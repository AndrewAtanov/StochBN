# StochBN
Experiments and StochBN implementation for pytorch.


# Experiments

*`validation_exp.py` -- main experiment, comparison test accuracy with different BN strategy
* `batch_avg.py` -- averaging test predictions through many batches with BN in training mode (mean and variance compute from batch).
* `train_collected_stats.py` -- train network and during training switch BN layers to test mode (use collected mean and variance)

# Results

* Test accuracy for ResNet18 on CIFAR10 with different BN strategy

<table>
  <tr>
    <th>mean \ variance</th>
    <th>vanilla</th>
    <th>mean</th>
    <th>random</th>
  </tr>
  <tr>
    <td>vanilla</td>
    <td>0.9407</td>
    <td>0.9407</td>
    <td>0.9422</td>
  </tr>
  <tr>
    <td>mean</td>
    <td>0.9407</td>
    <td>0.9407</td>
    <td>0.9412</td>
  </tr>
  <tr>
    <td>random</td>
    <td>0.9415</td>
    <td>0.9419</td>
    <td>0.9415</td>
  </tr>
</table>

# Acknowledgement
* Thanks https://github.com/kuangliu for models https://github.com/kuangliu/pytorch-cifar
