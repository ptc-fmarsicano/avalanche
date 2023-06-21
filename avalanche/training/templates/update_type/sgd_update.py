class SGDUpdate:
    def training_epoch(self, **kwargs):
        """Training epoch.

        :param kwargs:
        :return:
        """
        for (batch_iter, self.mbatch) in enumerate(self.dataloader):
            if self._stop_training:
                break
            assert self.mbatch[0].shape[0] == 64
            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = 0

            # Forward
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            self._after_forward(**kwargs)

            # Loss & Backward
            self.loss += self.criterion()

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            kwargs.update({"batch_iter": batch_iter})
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)
