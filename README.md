Various_softmax_by_Keras
======
Keras implementation of various softmax function  

****
	
|Author|LCorleone|
|---|---
|E-mail|645096158@qq.com


****
## Requirements
* tensorflow 1.5
* keras 2.2.0
* some common packages like numpy and so on.

## Quick start
* Now we support original softmax, center loss, A-softmax and AM-softmax.  
* Just change the loss_name in keras_mnist_loss_compare.py
```
loss_name = 'AM-softmax'
```

## Mnist Visualization
### original softmax
![original-softmax](https://github.com/LCorleone/Various_softmax_by_Keras/blob/master/gif/softmax.gif)
### center-loss
![center-loss](https://github.com/LCorleone/Various_softmax_by_Keras/blob/master/gif/center-loss.gif)
### A-softmax
![A-softmax](https://github.com/LCorleone/Various_softmax_by_Keras/blob/master/gif/A-softmax.gif)
### AM-softmax
![AM-softmax](https://github.com/LCorleone/Various_softmax_by_Keras/blob/master/gif/AM-softmax.gif)
## To do

- [x] original softmax
- [x] center loss
- [x] A-softmax
- [x] AM-softmax
- [ ] contranstive loss
- [ ] triplet loss

## Reference
[margin-softmax](https://github.com/bojone/margin-softmax) 
