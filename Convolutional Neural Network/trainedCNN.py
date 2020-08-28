import tensorflow as tf
tf.__version__

classifierLoad = tf.keras.models.load_model('model.h5')
classifierLoad.summary()


from keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/sp',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

result=classifierLoad.predict(test_set)
for i in range(0,len(result)):
    if result[i][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'
    print(prediction)
    
import numpy as np
from keras.preprocessing import image

##make multiple prediction using hardcoded loop
#pics=['1','2']
#
#for i in range(0,2):
#    test_image = image.load_img('dataset/sp/sp/'+pics[i]+'.jpg', target_size = (64, 64))
#    test_image = image.img_to_array(test_image)
#    test_image = np.expand_dims(test_image, axis = 0)
#    result=classifierLoad.predict(test_image)
#    if result[0][0] == 1:
#        prediction = 'dog'
#    else:
#        prediction = 'cat'
#    print(prediction)

## Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix, accuracy_score
#cm = confusion_matrix(test_set, result)
#print(cm)
#accuracy_score(test_set, result)

 