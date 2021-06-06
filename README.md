## Current steps

1. Run `prepare_data.py` to prepare data accordingly.
2. Train the GAN by running `train_gan.py`.
3. Make sure that the GAN was able to properly train by visualizing the distribution of the majority, minority, and some translated synthesized points using UMAP. A visualization snippet can be found in `visualize.py` or the `notebook.ipynb`. If the result does not look good when eyeballed, we can try modifying some of the following: the architectures of the generator/discriminators, the type of GAN loss, the hyperparameter in the GAN loss that penalizes distance.
4. Use the above GAN to synthesize minority points by translating from the majority samples, filtering them out according to the nearest neighbor criterion (we can try doubling or tripling the number of minority points and discarding the rest), and then running a classifier of our choosing on the dataset including the synthesized minority points (MLP, catboost, etc.). This is all done using `train_classifier.py`. Currently we're using catboost, with an MLP-based classifier commented out.
5. Compare the results of the classifier to a baseline classifier without synthesized samples.