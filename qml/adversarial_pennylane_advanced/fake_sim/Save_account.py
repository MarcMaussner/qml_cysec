from qiskit_ibm_runtime import QiskitRuntimeService

if __name__ == "__main__":
    # saving account
    print("Saving account")

 
    QiskitRuntimeService.save_account(
        token="vLrlVL9wpcpPenOg1AYB4R6aDxacYsHdo_4NvtTBX6ks", # Use the 44-character API_KEY you created and saved from the IBM Quantum Platform Home dashboard
        name="ibm_qgss", # Optional
        instance="crn:v1:bluemix:public:quantum-computing:us-east:a/cc574274bd254c1a92dbea97a7f4b564:bbe09c3f-19b8-41c9-88cd-9317af754997::", # Optional
        set_as_default=True, # Optional
        overwrite=True, # Optional
)