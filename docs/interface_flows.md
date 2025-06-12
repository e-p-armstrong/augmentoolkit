# Interface

The Interface is meant to be a powerful and easy way to use Augmentoolkit. It is the best way to learn, as the interface has been built to guide its user to the next most likely action from any given state, through the use of highlights, hint text, and emphasis. In the modern version of Augmentoolkit, the interface has stopped being a throwaway add-on that is only useful for the most basic of tasks -- it is now reimagined as a powerful tool to make datagen workflows more convinient, rather than just being a crutch for people who don't know how to use a CLI. The hope is that even if you do know how to use a command line, you might opt for the interface, because it is that much better.

It's also had some effort put into making it visually charming and fun to use.

To run the interface, you want to run one of the OS-specific start scripts, so that all the services are started. `macos.sh`, `linux.sh`, or `windows.bat`. You don't want to run `run_augmentoolkit.py` like you do with the CLI.

## Flows

### Upload documents

Uploading files can be done from the `Input` tab. Drag and drop, or click the Upload button on the top right and select a folder or file. When uploading a folder or file, the place it has been uploaded to will be indicated on the interface immediately after it is created.

Files can be moved around by clicking and dragging. Double clicking on a folder expands it.

All files uploaded via the interface end up in the `./inputs` folder of the Augmentoolkit project.

If you click on a folder and then click on the upload button, any files/folders uploaded will be uploaded to that folder.

### Starting runs

Navigate to a config file from the `Configs` tab. Double click on a file to open it. You can also delete config files.

When a config file is opened you can edit it. Saves are not automatic but the interface warns you if you have not saved. If any `!!PLACEHOLDER!!`s or ``!!ATTENTION!!`` markers are on the config (indicating things that you MUST change, or which are potential footguns from the default config file, respectively) then the edit interface will let you know about them. Once everything looks in place, the run button will turn green, and clicking on that will execute that config file's pipeline using those settings.

### Observing Runs

The Pipeline Monitor page (`Monitor`) allows you to view the logs of the pipeline execution as well as the overall progress on the progress bar. You can navigate away from this page and back to it, though be careful that if you clear the pipeline execution history you will not be able to monitor that run anymore. However since Augmentoolkit pipelines are resumable from interrupted execution, you can just run the same config that you had run at the start and you should be fine — the outputs will have almost certainly been saved to the 

### Getting your Results Back

This can be done in one of two ways. The output folder you specified in your config will appear in the `Outputs` tab, if that output folder is somewhere within `./outputs` within the Augmentoolkit project. You can also Download a run's output folder from the Pipeline monitor page once it finishes execution.

Admittedly, the Download button is almost more intended for potential future hosting. If you've git cloned this project and are running it locally, the advantage of "Downloading" files that already exist on your machine locally (effectively just makes a copy in your downloads folder) is probably low. This being the case, I want to note that you can also just find your pipeline's output folder in the `./outputs` folder of the Augmentoolkit project. However if you're very new to all of this and want to stay within the areas of your computer you're familiar with -- or if you want to have the Augmentoolkit API as a redundant safety layer between you and your output files -- then you can use the download buttons.

When you train models be sure to be [conscious of optimizer steps](../README.md#note-for-when-you-start-training)!