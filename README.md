## Pytorch-ALAE

## Original Repos and Difference 
Source: The ALAE part of this repo is copied from https://github.com/ariel415el/SimplePytorch-ALAE 

Difference: Added New Dataset for Anime Character.

Source: The FID part of this repo is copied from https://github.com/hukkelas/pytorch-frechet-inception-distance/tree/c4bef90e502e7e1aec2a1a8f45b259630e093f8b

Difference: Original Repo will crash if loading a large database since it read all images at once and they are stored in the RAM, the new implementation will read the path to get the images during calculation of the fid

Please checkout their github and thank you for their contribution

The anime dataset can be downloaded in here: https://drive.google.com/drive/folders/13H-8T0fbRxTT8SPE_5PoEG9hVUCe8sWO?usp=sharing Put the folder named Twdne-128 into data folder , them put another .pt dataset file inside the Twdne-128 Folder. Please checkout the following file structure.

The result after training for 100000 iterations is as below,

![text](assets/animeresult.png) 

The first row : Original Image
The second row : Reconstructed Image
The third row : Random Generated Image by Noise

## File Structure: 
	Lassests 
	Ldnn 
	... 
	Ldata 
        LTwdne-128 
            Ltwdne.pt
			
## Training: 
Remember to add argument when you run training: 

    python train_StyleALAE--dataset_name twdne

## Calculating FID
If you do not have two dataset for comparison, you need to generate a reconstructed one or a random generated one to compare with the original dataset.
Anyways, you can just use the test dataset in the repo.

![text](data/Test/original/00001.png) ![text](data/Test/reconstruct/00001.jpg) 

LHS is Original , RHS is Reconstructed Image

Command:

	python ./score_calculator/fid.py --path1 ../data/Test/original --path2 ../data/Test/reconstruct --batch-size 8

Result:

    Looking for images in ../data/Test/original/*.png
    Looking for images in ../data/Test/original/*.jpg
    Looking for images in ../data/Test/reconstruct/*.png
    Looking for images in ../data/Test/reconstruct/*.jpg
    Calculating Activation
     Progress: [------------------->] 100 %--Done--
    Calculating Activation
     Progress: [------------------->] 100 %--Done--
    The FID of these datasets :  133.45649989750808
    Finished

## Reconstructing or Generate Images using ALAE
If you want to reconstruct / generate image , you can run the test_StyleALAE file, but remember that you need to set up the pretrained model path:

-Select your model in the path

-Select the location of saving in the saved_path 

-Uncomment generate_img for generating image , at the current stage there is no saving function for generation (TODO)

-This part will input a batch size of 1 tensor into our model , which will be slow --> Will be changed to selectable bs (TODO)

-Remember to set the alpha value to a correct value , this info should be available on the name of your pt file 

    if __name__ == '__main__':
			path = "./data/FFHQ-thumbnails/thumbnails128x128"
			saved_path = "./data/FFHQ-thumbnails/reconstructed_thumbnails128x128"
			model = StyleALAE(model_config=config, device=device)
			model.load_train_state('pretrainedmodel.pt')

			batch_size = 32
			batchs_in_phase = config['phase_lengths'][model.res_idx] // batch_size
			alpha = 64/(config['phase_lengths'][model.res_idx])
			# generate_img(model, config, batch_size) 
			reconstruct_img(model, config, path, saved_path, False)

Command: 

	python test_StyleALAE

	