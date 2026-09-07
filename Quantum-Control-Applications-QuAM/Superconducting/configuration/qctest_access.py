from qctss_client.client import QCTSSClient
# %%
# Save TOKEN
my_token = ""
QCTSSClient.save_token(token=my_token)


# %%
# Ensure you have installed the SDK and set up your API key.
client = QCTSSClient()

job_response = client.start_job(
    qc_setup_list=["10FQ9FCv2#12_251222_DR4_OPX1000_4_0", "10FQ9FCv2#12_251222_DR4_OPX1000_4_1"],
    service_name="QPU Screening",
)
job_status = client.wait_until_running(job_response.job_id)
print(job_status.port_number)
# %%
# For Quantum Machines (QM) machines
from quam_libs.components import QuAM

machine = QuAM.load()
machine.network["cluster_name"] = "OPX1000_4"
machine.network["port"] = job_status.port_number
machine.save()

# %%