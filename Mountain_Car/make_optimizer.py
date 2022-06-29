from tensorflow_privacy.privacy.dp_query import gaussian_query
import tensorflow as tf

def make_keras_optimizer_class(cls):

  class DPOptimizerClass(cls):  # pylint: disable=empty-docstring
    __doc__ = "D".format(
        base_class='tf.keras.optimizers.' + cls.__name__,
        short_base_class=cls.__name__,
        dp_keras_class='DPKeras' + cls.__name__)

    def __init__(
        self,
        l2_norm_clip,
        noise_multiplier,
        num_microbatches=None,
        gradient_accumulation_steps=1,
        dp_flag=0,
        *args,  # pylint: disable=keyword-arg-before-vararg, g-doc-args
        **kwargs):
      super().__init__(*args, **kwargs)
      self.gradient_accumulation_steps = gradient_accumulation_steps
      self._l2_norm_clip = l2_norm_clip
      self._noise_multiplier = noise_multiplier
      self._num_microbatches = num_microbatches
      self._dp_sum_query = gaussian_query.GaussianSumQuery(
          l2_norm_clip, l2_norm_clip * noise_multiplier)
      self._global_state = None
      self._was_dp_gradients_called = False
      self.dp_flag = dp_flag

    def _create_slots(self, var_list):
      super()._create_slots(var_list)  # pytype: disable=attribute-error
      if self.gradient_accumulation_steps > 1:
        for var in var_list:
          self.add_slot(var, 'grad_acc')

    def _prepare_local(self, var_device, var_dtype, apply_state):
      super()._prepare_local(var_device, var_dtype, apply_state)  # pytype: disable=attribute-error
      if self.gradient_accumulation_steps > 1:
        apply_update = tf.math.equal(
            tf.math.floormod(self.iterations + 1,
                             self.gradient_accumulation_steps), 0)
        grad_scaler = tf.cast(1. / self.gradient_accumulation_steps, var_dtype)
        apply_state[(var_device, var_dtype)].update({
            'apply_update': apply_update,
            'grad_scaler': grad_scaler
        })

    def _resource_apply_dense(self, grad, var, apply_state=None):
      if self.gradient_accumulation_steps > 1:
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                        self._fallback_apply_state(var_device, var_dtype))
        grad_acc = self.get_slot(var, 'grad_acc')

        def _update_grad():
          apply_grad_op = super(DPOptimizerClass, self)._resource_apply_dense(
              grad_acc + grad * coefficients['grad_scaler'], var, apply_state)  # pytype: disable=attribute-error
          with tf.control_dependencies([apply_grad_op]):
            return grad_acc.assign(
                tf.zeros_like(grad_acc),
                use_locking=self._use_locking,
                read_value=False)

        def _accumulate():
          return grad_acc.assign_add(
              grad * coefficients['grad_scaler'],
              use_locking=self._use_locking,
              read_value=False)

        return tf.cond(coefficients['apply_update'], _update_grad, _accumulate)
      else:
        return super()._resource_apply_dense(grad, var, apply_state)  # pytype: disable=attribute-error

    def _resource_apply_sparse_duplicate_indices(self, *args, **kwargs):
      if self.gradient_accumulation_steps > 1:
        raise NotImplementedError(
            'Sparse gradients are not supported with large batch emulation.')
      else:
        return super()._resource_apply_sparse_duplicate_indices(*args, **kwargs)  # pytype: disable=attribute-error

    def _resource_apply_sparse(self, *args, **kwargs):
      if self.gradient_accumulation_steps > 1:
        raise NotImplementedError(
            'Sparse gradients are not supported with large batch emulation.')
      else:
        return super()._resource_apply_sparse(*args, **kwargs)  # pytype: disable=attribute-error

    def _compute_gradients(self, loss, var_list, grad_loss=None, tape=None):
      """DP-SGD version of base class method."""

      self._was_dp_gradients_called = True
      # Compute loss.
      if not callable(loss) and tape is None:
        raise ValueError('`tape` is required when a `Tensor` loss is passed.')
      tape = tape if tape is not None else tf.GradientTape()

      if callable(loss):
        with tape:
          if not callable(var_list):
            tape.watch(var_list)

          loss = loss()
          if self._num_microbatches is None:
            num_microbatches = tf.shape(input=loss)[0]
          else:
            num_microbatches = self._num_microbatches
          microbatch_losses = tf.reduce_mean(
              tf.reshape(loss, [num_microbatches, -1]), axis=1)

          if callable(var_list):
            var_list = var_list()
      else:
        with tape:
          if self._num_microbatches is None:
            num_microbatches = tf.shape(input=loss)[0]
          else:
            num_microbatches = self._num_microbatches
          microbatch_losses = tf.reduce_mean(
              tf.reshape(loss, [num_microbatches, -1]), axis=1)

      var_list = tf.nest.flatten(var_list)

      # Compute the per-microbatch losses using helpful jacobian method.
      with tf.keras.backend.name_scope(self._name + '/gradients'):
        jacobian = tape.jacobian(
            microbatch_losses, var_list, unconnected_gradients='zero')

        # Clip gradients to given l2_norm_clip.
        def clip_gradients(g):
            if self.dp_flag == 0:
                return tf.clip_by_global_norm(g, 100000)[0]
            else:
                return tf.clip_by_global_norm(g, self._l2_norm_clip)[0]

        clipped_gradients = tf.map_fn(clip_gradients, jacobian)

        def reduce_noise_normalize_batch(g):
          # Sum gradients over all microbatches.
          summed_gradient = tf.reduce_sum(g, axis=0)

          if self.dp_flag == 0:
              noised_gradient = summed_gradient
          else:
            # Add noise to summed gradients.
            noise_stddev = self._l2_norm_clip * self._noise_multiplier
            noise = tf.random.normal(
                tf.shape(input=summed_gradient), stddev=noise_stddev)
            noised_gradient = tf.add(summed_gradient, noise)

          # Normalize by number of microbatches and return.
          return tf.truediv(noised_gradient,
                            tf.cast(num_microbatches, tf.float32))

        final_gradients = tf.nest.map_structure(reduce_noise_normalize_batch,
                                                clipped_gradients)

      return list(zip(final_gradients, var_list))

    def get_gradients(self, loss, params):
      """DP-SGD version of base class method."""

      self._was_dp_gradients_called = True
      if self._global_state is None:
        self._global_state = self._dp_sum_query.initial_global_state()

      # This code mostly follows the logic in the original DPOptimizerClass
      # in dp_optimizer.py, except that this returns only the gradients,
      # not the gradients and variables.
      microbatch_losses = tf.reshape(loss, [self._num_microbatches, -1])
      sample_params = (
          self._dp_sum_query.derive_sample_params(self._global_state))

      def process_microbatch(i, sample_state):
        """Process one microbatch (record) with privacy helper."""
        mean_loss = tf.reduce_mean(
            input_tensor=tf.gather(microbatch_losses, [i]))
        grads = tf.gradients(mean_loss, params)
        sample_state = self._dp_sum_query.accumulate_record(
            sample_params, sample_state, grads)
        return sample_state

      sample_state = self._dp_sum_query.initial_sample_state(params)
      for idx in range(self._num_microbatches):
        sample_state = process_microbatch(idx, sample_state)
      grad_sums, self._global_state, _ = (
          self._dp_sum_query.get_noised_result(sample_state,
                                               self._global_state))

      def normalize(v):
        try:
          return tf.truediv(v, tf.cast(self._num_microbatches, tf.float32))
        except TypeError:
          return None

      final_grads = tf.nest.map_structure(normalize, grad_sums)

      return final_grads

    def get_config(self):
      config = super().get_config()
      config.update({
          'l2_norm_clip': self._l2_norm_clip,
          'noise_multiplier': self._noise_multiplier,
          'num_microbatches': self._num_microbatches,
      })
      return config

    def apply_gradients(self, *args, **kwargs):
      return super().apply_gradients(*args, **kwargs)

  return DPOptimizerClass