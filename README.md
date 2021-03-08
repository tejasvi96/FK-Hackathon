# FK-Hackathon
Flipkart hackathon submission

To do inference set the imagesdir parameter which should point to the directory where all the images for testing are present, set the testnames flag also to refer to the .csv file for reading the filenames for generating predictions. The output file will be generated based on the value of the output_file parameter.
 
``` pip install -r requirements.txt ``` 

to install all the dependencies.

To use the code for inference

```python demo_fk_gcn.py -t --resume './checkpoint_modified42.pth' --imagesdir "./test_images" --testnames "./extra/Test_Filenames.csv" --output_file "./outputs.csv" ```

If cuda is available , then by default it will run on gpu, if not then on cpu.


# References

We took reference from the paper [Multi-Label Image Recognition with Graph Convolutional Networks](https://arxiv.org/abs/1904.03582) for the multi label classification and adapted their code from [here](https://github.com/Megvii-Nanjing/ML-GCN) to our problem.

