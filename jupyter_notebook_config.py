from IPython.lib import passwd

c.NotebookApp.iopub_data_rate_limit = 1000000000
c.NotebookApp.ip = "*"
c.NotebookApp.allow_root = True
c.NotebookApp.password = passwd("dl4img")

