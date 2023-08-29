## On the computational and parameter efficiency of Low Rank Neural Networks

Code supplement for all experiments of the Bachelor's thesis.
This repository started as a copy of the original repository for below mentioned paper, which isn't publicly available anymore. I added the following files for my research:
 - TestMatrix.py (main testing script, which is able to run a lot of different parameters)
 - evaluation.py (used to generate most of the graphs in the thesis)
 - evaluationMultiple.py (used to generate graphs with multiple tests like Figure 3.11)
 - evaluationRain.py (used to generate Figure 3.13)
 - evaluationRainFullSet.py (used to generate Figure 3.12)
 - further_graphics.py (used to generate Figures 1.1, 1.2 and 2.4)
 - rename.py (used to rename some data)

Additionally I made some minor changes to "mnist_DLRT.py", "mnist_DLRT_fr.py", "mnist_reference.py" and "dense_layers.py" to make some of my experiments possible. For example I added the possibility of regularization.

Furthermore I added the folders "finishedTests", which contains raw data for all tests with a dataset size of 5K-1K, and "finishedTestsFullSet", which contains data for all tests with a dataset size of 60K-10K. "wrongTests" contains only unusable data.
The naming of the test cases was done by adding different text parts together dependent on the parameters used:

	method		    	descent algorithm		dimensions		epochs		batch size		learning rate		noise		dataset-size		regularization

	"DLRT(Rank[x-y])"   	"SGD"				"TestDim[x-y]"		"ep[x]"		"batch[x]"		"lr[x]"			"Noise[x]"	"Dataset[x]"		"L1L2Reg[x]"
	""			""															""		""			"L1Reg[x]"
																								"Reg[x]"
																								"RegFull[x]"
																								""

"" for method means using the reference network and for descent algorithm Adam. "" for noise means applying no noise. "" for dataset-size means 60K-10k and "Dataset10" means 5K-1K. "Reg[x]" for regularization means L2-regularization
with lambda x in percent, where 0-1 means 0.001. "" means no regularization.

Inside the runs we find "historyLogs", which contains the actual data. Each file in this folder stands for one dimension (or rank, if we test multiple ranks). Each row in the excel file is one epoch.






## Original ReadMe:
## [Low-rank lottery tickets: finding efficient low-rank neural networks via matrix differential equations](https://arxiv.org/abs/2205.13571)

Code supplement for all dense neural network experiments of the arXiv Preprint

### Usage

1. create a python virtual environment (pyenv or conda). If you are using no virtual environment, please be aware of
   version incompatibilities of tensorflow
2. Install the project requirements (example for pip):
   ``pip install -r requirements.txt``
3. Run the batch scripts for the test cases
    1. ``sh run_tests_dense_reference.sh`` trains a baseline traditional dense network.
    2. ``sh run_tests_fixed_rank.sh`` trains a fixed low-rank network using our proposed method. This method is used in
       section 5.1
    3. ``sh run_tests_rank_adaptive.sh`` trains a network using our proposed rank adaptive algorithm to find optimal low
       ranks
    4. ``sh run_tests_fixed_rank_fine_tuning.sh`` trains a network using our proposed rank adaptive algorithm to find
       optimal low ranks. Once the low-ranks are found, the script switches to dynamical fixed rank training to
       fine-tune the model. This method is used in Section 5.2
    4. ``sh run_tests_fixed_rank_train_from_prune.sh`` loads the weights of a traditional network (provided in the
       folder "dense_weights"), then factorizes the weight matrix and truncates all but 20 eigenvalues. Then, fixed
       low-rank training is used to retrain the model. This method is used in Section 7.3
    5. ``sh run_test_transformer_dlrt.sh`` and ``sh run_test_transformer_big_dlrt.sh`` trains a transformer on the
       portuguese to english translation task with DLRT
    6. ``sh run_test_transformer_fix_rank.sh`` and ``sh run_test_transformer_big_fix_rank.sh`` trains a transformer on
       the portuguese to english translation task with fixed rank DLRT.
     
### Useful links

The pytorch version can be found [here](https://github.com/COMPiLELab/DLRT/tree/efficient_gradient)

