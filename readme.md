# TFT Training module
This is a brief documentation of the TFT data processing and training module used in the “Stk.” platform.

## Cloud Scheduler
The automatic training and data processing process is achieved by using google cloud scheduler to train on a cron job schedule. Working with the Pub/Sub feature of the google cloud platform, we schedule training as a google cloud function to fetch and process the data at a specific time, then train and upload the model weights to a cloud blob for use in the prediction module.


```bash
import base64
from googleapiclient import discovery
import pytz
import datetime

	#-------------------- Configurations --------------------
GCP_PROJECT = "thesis-stk-project"
GCS_BUCKET_PATH = "gs://thesis-stk-project"
STARTUP_SCRIPT_URL = "https://storage.googleapis.com/35f1713a93cb4001-dot-us-west1.notebooks.googleusercontent.sh" 
	

PROJECT_NAME = " spartan-matter-310516"
NOTEBOOK_NAME = "ol.ipynb"
DLVM_IMAGE_PROJECT = "deeplearning-platform-release"
DLVM_IMAGE_FAMILY = "tf2-2-0-cu100"
ZONE = "us-west1-b"
MACHINE_TYPE = "n1-highmem-8"
MACHINE_NAME = PROJECT_NAME
BOOT_DISK_SIZE = "200GB"
GPU_TYPE = "nvidia-tesla-k80"
GPU_COUNT = 1
INSTALL_NVIDIA_DRIVER = True


	

def create_instance():
	

	    # Create the Cloud Compute Engine service object
	    compute = discovery.build('compute', 'v1')
	    
	    image_response = compute.images().getFromFamily(
	        project=DLVM_IMAGE_PROJECT, family=DLVM_IMAGE_FAMILY).execute()
	    source_disk_image = image_response['selfLink']
	

	    # Configure the machine
	    machine_type_with_zone = "zones/%s/machineTypes/%s" % (ZONE,MACHINE_TYPE)
	    
	    today_date = datetime.datetime.now(pytz.timezone('Asia/Jakarta'))
	

	    gcs_input_notebook = "%s/notebooks/%s/current/%s" % (GCS_BUCKET_PATH,PROJECT_NAME,NOTEBOOK_NAME)
	    gcs_output_folder = "%s/outputs/%s/%s/%s/%s/" % (GCS_BUCKET_PATH,PROJECT_NAME,today_date.year,today_date.month,today_date.day)
	    gcs_parameters_file= "%s/notebooks/%s/current/%s" % (GCS_BUCKET_PATH,PROJECT_NAME,"params.yaml")
	    gcs_requirements_txt= "%s/notebooks/%s/current/%s" % (GCS_BUCKET_PATH,PROJECT_NAME,"requirements.txt")
	

	    accelerator_type = "projects/%s/zones/%s/acceleratorTypes/%s" % (GCP_PROJECT,ZONE,GPU_TYPE)
	

	    config = {
	        'name': MACHINE_NAME,
	        'machineType': machine_type_with_zone,
	

	        # Specify the boot disk and the image to use as a source.
	        'disks': [
	            {
	                'boot': True,
	                'autoDelete': True,
	                'initializeParams': {
	                    'sourceImage': source_disk_image,
	                },
	                'boot-disk-size': BOOT_DISK_SIZE
	            }
	        ],
	        
	        # Specify a network interface with NAT to access the public
	        # internet.
	        'networkInterfaces': [{
	            'network': 'global/networks/default',
	            'accessConfigs': [
	                {'type': 'ONE_TO_ONE_NAT', 'name': 'External NAT'}
	            ]
	        }],
	

	        'guestAccelerators': [{
	            'acceleratorType':accelerator_type,
	            'acceleratorCount':GPU_COUNT
	        }],
	

	        'scheduling': {
	            'onHostMaintenance': 'TERMINATE'
	        },
	

	        # Allow the instance to access cloud storage and logging.
	        'serviceAccounts': [{
	            'email': 'default',
	            'scopes': [
	                'https://www.googleapis.com/auth/cloud-platform'
	            ]
	        }],
	

	        # Metadata is readable from the instance and allows you to
	        # pass configuration from deployment scripts to instances.
	        'metadata': {
	            'items': [{
	                # Startup script is automatically executed by the
	                # instance upon startup.
	                'key': 'startup-script-url',
	                'value': STARTUP_SCRIPT_URL
	            }, {
	                'key': 'input_notebook',
	                'value': gcs_input_notebook
	            }, {
	                'key': 'output_notebook',
	                'value': gcs_output_folder
	            }, {
	                'key': 'requirements_txt',
	                'value': gcs_requirements_txt 
	            }, {
	                'key': 'parameters_file',
	                'value': gcs_parameters_file
	            }, {
	                'key': 'install-nvidia-driver',
	                'value': INSTALL_NVIDIA_DRIVER
	            }]
	        }
	    }
	

	   return compute.instances().insert(
	       project=GCP_PROJECT,
	       zone=ZONE,
	       body=config).execute()
```

This is the main module in the Pub/Sub part of the Google Cloud platform which enables the cron job on the Cloud Scheduler to trigger the training process on the Google cloud function, which does the actual pre-processing and prediction.

![Scheduler](https://drive.google.com/file/d/1iDPS2W4TTdK6J86z7sDp1QOuQpQNoiSU/view?usp=sharing)


## Online Preprocessing Module 
To be able to use newly collected data, preprocessing needs to be done daily as more data is collected. This is outlined below.
### Special Imports
below are some special dependencies and their explanations
```bash
!pip install git+https://github.com/awael/pytorch-lightning@cpytorch
# modified pytorch to fit requirements and changes of the project
!pip install pytorch-forecasting==0.8.4
#stable version
!pip install gensim==3.6.0
#stable version
!pip install tensorflow==2.4.1
#stable version
```
### Fetching Data
to be able to transfer data from the CGP memory blob (used in storing model weights, preprocessed datasets, etc.) we define the function below:
```bash
from google.cloud import storage


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )
```
### API Exchange
to fetch the data from the AWS server database, we set up the following API call which is explained in the website documentation:
```bash
dd = pd.read_excel('http://3.65.72.21:8080/downloadData')
```
### Data Reshaping
#### main steps of data reshaping include:
Sorting data by date:
```bash
dd = dd.sort_values(by=['isodate'])
```
Collecting by news outlet:
```bash
dd['alborsaanews'] = dd.groupby(['isodate'])['alborsaanews'].transform(lambda x : ' '.join(x))
dd['almalnews'] = dd.groupby(['isodate'])['almalnews'].transform(lambda x : ' '.join(x))
dd['amwalalghad'] = dd.groupby(['isodate'])['amwalalghad'].transform(lambda x : ' '.join(x))
dd['mubasher'] = dd.groupby(['isodate'])['mubasher'].transform(lambda x : ' '.join(x))
```
Downloading Financial Ratios:
```bash
download_blob("stk-bucket-thesis","list.xlsx","list.xlsx")
dl = pd.read_excel('list.xlsx')
```

Downloading previously processed data:
```bash
download_blob("stk-bucket-thesis","EDITEDNEWDATA.xlsx","EDITEDNEWDATA.xlsx")
dr = pd.read_excel('EDITEDNEWDATA.xlsx')
```
Merging into one dataset:
```bash
finalratios = pd.merge(list2,drl,how='left',on = 'DATE')
```
### Doc2Vec module
after reformatting the data, we need to embed the scraped data using our doc2vec weights.
#### Download and load doc2vec weights:
```bash
from gensim.models.doc2vec import Doc2Vec
download_blob("stk-bucket-thesis","doc2vec/doc2vecModel_byday_onesource.bin","home/jupyter/doc2vec/doc2vecModel_byday_onesource.bin")
download_blob("stk-bucket-thesis","doc2vec/doc2vecModel_byday_onesource.bin.trainables.syn1neg.npy","home/jupyter/doc2vec/doc2vecModel_byday_onesource.bin.trainables.syn1neg.npy")
download_blob("stk-bucket-thesis","doc2vec/doc2vecModel_byday_onesource.bin.wv.vectors.npy","home/jupyter/doc2vec/doc2vecModel_byday_onesource.bin.wv.vectors.npy")

dvmodel = Doc2Vec.load("doc2vec/doc2vecModel_byday_onesource.bin")#model weights

```
after finishing preprocessing, we can now embed the text:
```bash
for i in tqdm(range(len(dd["alborsaanews"]))):
    new_alborsaanews.append(dvmodel.infer_vector(gensim.utils.simple_preprocess(dd["alborsaanews"][i])))
    new_almalnews.append(dvmodel.infer_vector(gensim.utils.simple_preprocess(dd["almalnews"][i])))
    new_amwalalghad.append(dvmodel.infer_vector(gensim.utils.simple_preprocess(dd["amwalalghad"][i])))
    new_mubasher.append(dvmodel.infer_vector(gensim.utils.simple_preprocess(dd["mubasher"][i])))

```
after parsing and feature generation (remaining days, day of week, etc.) we merge all old and new data into one dataset.
#### merging all data:
```bash
merged_inner = pd.merge(left=fff, right=ddf, left_on='time_idx', right_on='time_idx')
```
## Training Parameters

Before starting training we define out time varying known reals/ categoricals and static reals and categoricals.
```bash
tvkr+=list(dd.columns[-9:])
```
### Training Setup
here we set up training parameters using the `TimeSeriesDataSet()` function:
```bash
training = TimeSeriesDataSet(
    dd[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="closing_price_val",
    group_ids=["company"],
    min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=28,
    max_prediction_length=max_prediction_length,
    static_categoricals=staticCategoricals,
    static_reals=staticReals,
    time_varying_known_categoricals=["year","month","day"],
    variable_groups=[],
    time_varying_known_reals=tvkr,
    time_varying_unknown_categoricals=tvuc,
    time_varying_unknown_reals=[
        "closing_price_val",
        "closing_price_pct",
        "log"
    ],
    target_normalizer=GroupNormalizer(
        groups=["company"], transformation="softplus"
    ),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)
```
### Validation
used for validation in non-live version
```bash
# create validation set (predict=True) which means to predict the last max_prediction_length points in time for each series
#validation = TimeSeriesDataSet.from_dataset(training, dd, predict=True, stop_randomization=True)
```
we also set a baseline:
```bash
actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
baseline_predictions = Baseline().predict(val_dataloader)
(actuals - baseline_predictions).abs().mean().item()
```
### Dataloaders
we then create dataloaders for the model:
```bash
# create dataloaders for model
batch_size = 32  # between 32 to 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
```
### Defining custom metrics
here, we define some metrics and make some changes to the PyTorch forecasting module to accommodate changes such as the normalized quantile loss function used. This is a sample of the changes:
```bash
class CustomError(MultiHorizonMetric):
    """
    Quantile loss, i.e. a quantile of ``q=0.5`` will give half of the mean absolute error as it is calcualted as
    Defined as ``max(q * (y-y_pred), (1-q) * (y_pred-y))``
    """

    def __init__(
        self,
        quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
        **kwargs,
    ):
        """
        Quantile loss
        Args:
            quantiles: quantiles for metric
        """
        super().__init__(quantiles=quantiles, **kwargs)

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # calculate quantile loss
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = (target - y_pred[..., i]) / target
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        losses = torch.cat(losses, dim=2)

        return losses

```
### Define the trainer
here, we define the trainer module, using a random seed and some hyperparameters.
```bash
pl.seed_everything(55)
trainer = pl.Trainer(
	gpus='0',
    # clipping gradients is a hyperparameter and important to prevent divergance
    # of the gradient for recurrent neural networks
    gradient_clip_val=0.2678493827151133,
)

```
### Define the TFT model
```bash
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.003721612229221799,
    hidden_size=80,  #important
    attention_head_size=4, # number of attention heads.
    dropout=0.146281009968505,  # between 0.1 and 0.3 are good values
    hidden_continuous_size=58,
    output_size=7,  #default
    loss=CustomError(),
    reduce_on_plateau_patience=4,    # reduces learning rate if no improvement
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
```
### Find optimal learning rate
used only when testing and not on online version, which uses the tested parameters.
```bash
# find optimal learning rate
res = trainer.tuner.lr_find(
    tft,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader,
    max_lr=10.0,
    min_lr=1e-6,
)

print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()

```
### Configuring the trainer
```bash
# configure network and trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs=100,
    gpus='0',
    weights_summary="top",
    gradient_clip_val=0.2678493827151133,
    #limit_train_batches=30,
    #fast_dev_run=True,
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)
```

### Start training
call `.fit()` to start training
```bash
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

# fit network
trainer.fit(
    tft,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader,
)
```
### Determine best checkpoint
```bash
# load the best model according to the validation loss
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

```
### Plots and metrics (non-live)
a sample of plots and tests performed:

#### plot best/worst scoring examples
```bash
predictions = best_tft.predict(val_dataloader)
mean_losses = SMAPE(reduction="none")(predictions, actuals).mean(1)
indices = mean_losses.argsort(descending=False)  # sort losses
for idx in range(10):  # plot 10 examples
    best_tft.plot_prediction(x, raw_predictions, idx=indices[idx]);
```
#### Mean absolute error
```bash
actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = best_tft.predict(val_dataloader)
(actuals - predictions).abs().mean()
```
## Saving model weights

### saving "locally" to the function :
```bash
trainer.save_checkpoint("./TFTol.ckpt")
```
### Upload to blob
#### `uplod_blob` function:
```bash
trainer.save_checkpoint("./TFTol.ckpt")
def upload_blob(bucket_name, source_file_name, destination_blob_name):
        """Uploads a file to the bucket."""
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)

        print(
            "File {} uploaded to {}.".format(source_file_name, destination_blob_name))
```
#### Uploading the model weights
```bash
upload_blob("stk-bucket-thesis","./TFTol.ckpt","TFTol.ckpt")
```
these uploaded weights are then downloaded and used in the prediction module. These are done separately in order to decouple training, which could be more realistically done every couple of days, with prediction which needs to run on a daily basis. 
