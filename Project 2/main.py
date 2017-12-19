"""Main file to run"""

#imports
import road.realdata.data as data
import road.realhelper.mask_to_submission as mask2sub
import road.realmodel.neural_network as NeuralNet
#load the data
#data = Data.Data()

#we preprocess the images

img_path = "training/images/"
gt_path = "training/groundtruth/"

preprocessed_img_path = "training/images/"
preprocessed_gt_path = "training/groundtruth/"

precomputed_model = "saved models/model 4.h5"
load_model = True
data_augmentation = True
patch_size = 8

#TODO créer cette fonction dans un package
def preprocess_imgs(a,b,c,d):
    pass

print("* started preprocessing")
#create preprocessed imgs into a new path
preprocess_imgs(img_path, preprocessed_img_path, gt_path, preprocessed_gt_path)

print("* finished preprocessing")
print("* loading preprocessed images")
training_patch_x, training_patch_y = data.create_xy_from_patch(preprocessed_img_path, preprocessed_gt_path, patch_size)

print("    patches X of shape: "+str(training_patch_x.shape))
print("    patches Y of shape: "+str(training_patch_y.shape))

print("* initialising model")



#choose if we want to load a nn model from an existing file, or create a new model using training on the patchs
if load_model:
    print("* loadîng model")
    model = NeuralNet.from_file(precomputed_model)
    print("* finished loading")
else:
    print("* start the training of the model")
    model = NeuralNet.create_model(training_patch_x, training_patch_y, data_augmentation)
    model.save('yolo.h5')
print(" * model summary")
model.summary()

id = 5
NeuralNet.predict_one_img(preprocessed_img_path, id)

test_path = "test_set_images/"
submission_name = 'model_d' 
test_set_results = "test_set_results_"+submission_name+"/"

print("* start predictions on the test set")
NeuralNet.predict_and_save_test_imgs(model,test_path,test_set_results, patch_size)
print("* finished prediction")
#now that we have the final images, we create the submission file
print("* creating submission file")
mask2sub.run(submission_name+".csv", "test_set_results_"+submission_name+"/")

print("* Finished")
