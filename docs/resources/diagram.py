from diagrams import Cluster, Diagram, Edge
from diagrams.custom import Custom
from diagrams.gcp.operations import Monitoring
from diagrams.oci.governance import Audit
from diagrams.oci.monitoring import Telemetry
from diagrams.oci.storage import ObjectStorage
from diagrams.onprem.queue import Kafka

with Diagram(name="Livesteram Data Reduction System Overview", show=False):
    with Cluster("Instrument VM"):
        ps_control = Monitoring("Process Controller \n(backend)")
        config = Audit("Configuration")

        with Cluster("visualization"):
            web_browser = ObjectStorage("Web Browser")
            dash_builder = Telemetry("Dashboard Builder")

            with Cluster("Data Handling"):
                with Cluster("Data Reduction Workflow"):
                    data_reductions = [
                        Custom("Sub Workflow 3", "./docs/resources/logo.png"),
                        Custom("Sub Workflow 2", "./docs/resources/logo.png"),
                        Custom("Sub Workflow 1", "./docs/resources/logo.png"),
                    ]

            dash_builder >> Edge() << data_reductions
            dash_builder >> web_browser

        kaf_listener = Kafka("Listener")
        kaf_listener >> dash_builder

    kaf_producer = Kafka("Producer")
    kaf_producer << Edge(label="subscribe") << kaf_listener
    web_browser >> ps_control >> web_browser
    dash_builder << ps_control
