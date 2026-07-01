# DREAM demo

## Setup

### Packages

```sh
python -m venv dream-demo
source dream-demo/bin/activate
pip install "git+https://github.com/scipp/esslivedata.git@main#egg=esslivedata[dream,dashboard]"
```

### Data

```sh
scp login.esss.dk:'/dmsc/scipp/dream/268227_00024779_*' .
```

## Running

### Docker

Clone the repository (main branch).
I usually run the Docker containers in a detached screen session, so I can reattach later if needed, but this is optional.

```sh
screen -S kafka
LIVEDATA_INSTRUMENT=dream docker compose -f docker-compose.yml up
```

Press `Ctrl + A` then `D` to detach from the screen session.
You can later reattach to the session with:

```sh
screen -r kafka
```

### ESSlivedata

In individual terminal windows, run (make sure to `source dream-demo/bin/activate` in each):

```sh
python -m ess.livedata.services.fake_detectors --instrument dream --nexus-file 268227_00024779_Si_BC_offset_240_deg_wlgth.hdf
python -m ess.livedata.services.data_reduction --instrument=dream --dev
python -m ess.livedata.dashboard.reduction --instrument=dream
```

You can also run the monitor-related services:

```sh
python -m ess.livedata.services.fake_monitors --instrument dream --mode da00
python -m ess.livedata.services.monitor_data --instrument=dream --dev
```

## Usage

### Connecting

Navigate to `http://localhost:5009` for the dashboard.
IDEs such as VScode with remote extensions will automatically forward this port to the host machine if you click on the link in the terminal when starting the dashboard.

### Data reduction

Try to configure and start workflows.
It usually takes a couple of seconds since static parts of the workflow (such as instrument geometry from NeXus files) have to be computed initially.

The data rate is set to something very low, so you can actually see changes in the plots.
The fake data source loops over the same events from the data file indefinitely.
We can easily deal with much higher rates.

Some config errors are caught and displayed in the UI, but if the backend reduction workflow fails you will not see an error in the UI.
You can check the logs in the terminal where you started the `data_reduction` service.

The streamed data is using choppers settings with the `high_flux_BC240` settings.
You can see/demonstrate artifacts from (incorrect) WFM frame merging by selecting `high_flux_BC215`.

#### Known issues

- The `logarithmic` bin settings work in the backend, but the UI plotting cannot deal with it.
- Changing units of edges probably does not work everywhere either.
- There is no indication of stopped workflows in the plots.

### Monitors

The fake monitors are not based on a simulation but simply show a Gaussian distribution of values around a mean value.

#### Known issues

- Pan/Zoom state resets with every update.

## Troubleshooting

ESSlivedata persists some config in Kafka.
In case this is causing hard-to-debug issues, you can clear the Kafka topics.
First, cancel all the services, otherwise some topics are automatically recreated (with incorrect settings).
Then run:

```sh
docker exec -it kafka-broker sh /scripts/setup-kafka-topics.sh dream
```

Alternatively, just `screen -r kafka`, `Ctrl + C` the entire thing, then relaunch.

You will most likely have to restart the services as they are not designed to handle topic removal gracefully.
