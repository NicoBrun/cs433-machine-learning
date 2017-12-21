"""Main file to run"""

#imports
import road.data.data as data
import road.helper.mask_to_submission as mask2sub
import road.model.neural_network as NeuralNet
import road.image_procesing.PostProcess as postProc

#we preprocess the images

img_path = "training/images/"
gt_path = "training/groundtruth/"

precomputed_model = "saved models/model_final.h5"
load_model = True
data_augmentation = True
patch_size = 16

test_path = "test_set_images/"
submission_name = 'model_final' 

post_process = True

print("* loading images")
training_patch_x, training_patch_y = data.create_xy_from_patch(img_path, gt_path, patch_size)

print("    patches X of shape: "+str(training_patch_x.shape))
print("    patches Y of shape: "+str(training_patch_y.shape))

print("* initialising model")



#choose if we want to load a nn model from an existing file, or create a new model using training on the patchs
if load_model:
    print("* loading model")
    model = NeuralNet.from_file(precomputed_model)
    print("* finished loading")
    print(" * model summary")
    model.summary()
else:
    print("* start the training of the model")
    model = NeuralNet.create_model(training_patch_x, training_patch_y, data_augmentation, patch_size)
    model.save(submission_name+'.h5')



test_set_results = "test_set_results_"+submission_name+"/"
post_process_test_results = "post_process_"+submission_name+"/"

if(not post_process) :
    post_process_test_results = test_set_results


#predict the image
print("* start predictions on the test set")
NeuralNet.predict_and_save_test_imgs(model, test_path, test_set_results, patch_size)
print("* finished prediction")

if(post_process):
    print("*start post-processing")
    postProc.post_process(test_set_results,post_process_test_results)
    print("*end of post-processing")
#now that we have the final images, we create the submission file
print("* creating submission file")
mask2sub.run(submission_name+".csv", post_process_test_results)

print("* Finished")
