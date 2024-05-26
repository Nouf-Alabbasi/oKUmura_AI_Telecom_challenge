import requests
import json

# Replace 'your-api-url' with your actual API Gateway URL
url = 'https://a7tncd274b.execute-api.eu-west-3.amazonaws.com/InitialStage/Falcon7B'


# "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",

# Character Count: 6632
# Estimated Token Count: 1658

question_184 = '''
### Instruction:
    Please provide the answers to the following multiple choice question.
   How can cloud computing benefit context management in the IoT (Internet of Things)?


   
### context:
Considering the following context:

Terms and Definitions:




Abbreviations:

IoT: Internet of Things



    Considering the following context:

Retrieval 1:
...to a target eNB using the X2 interface, the source eNB should propagate the trace control and configuration parameters further to the target eNB by using the HANDOVER REQUEST message. When the target eNB receives the HANDOVER REQUEST message it should immediately start a Trace Session according to the trace control and configuration parameters received in the HANDOVER REQUEST message.
If the subscriber or equipment which is traced makes a handover to a target eNB using the S1 interface, it is the ...
This retrieval is performed from the document 32422-i10.docx.


Retrieval 2:
...to EPS handover using N26
0.	A PDU session is established in 5GS with multiple QoS Flows. A Charging Identifier  was assigned to the PDU session.
0ch. A charging session between the PGW-C+SMF and CHF exists for this PDU session.
10c. PDU session update response to AMF.
10ch-a. This step occurs if steps 10a-c occurred. All counts are closed and a Charging Data Request [Update] is sent to CHF, if required by "Handover start" trigger. New counts and time stamps for all active service data flows are ...
This retrieval is performed from the document 32255-i20.docx.


Retrieval 3:
...Switched domain: MSC address and Call Reference Number;
-	Packet Switched domain: P-GW address and EPC Charging ID;
-	5G Data connectivity domain: 5GC Charging ID;
-	Fixed Broadband Access: Multimedia Charging ID;
-	IM Subsystem: IMS Charging Identifier.
The charging information has to be aggregate for the same charging session and correlate for the same service.
5.3.4.1	Intra-level correlation
The intra-level correlation aggregates the charging events belonging to the same charging session, e.g. ...
This retrieval is performed from the document 32240-i50.docx.


Retrieval 4:
...session transfer.
5. The S-CSCF forwards the SIP INVITE to the SCC Application over the ISC interface.
6 The SCC AS analyses the SIP INVITE to derive that the SIP INVITE is a request to transfer a session from UE#1 to UE#2.
7-9. The SCC AS sends a SIP Re-INVITE to the Remote UE to update the media components of the previous dialog.  Remote UE answers by a SIP 200 OK message.
10-12. At Remote Leg update the S-CSCF in the originating network sends Charging Data Request[Interim] to record update of ...
This retrieval is performed from the document 32260-i20.docx.


Retrieval 5:
...reporting from RAN of downlink data volumes is optional.
-	RAN Start Time is a time stamp, which defines the moment when the volume container is opened by the RAN.
-	RAN End Time is a time stamp, which defines the moment when the volume container is closed by the RAN.
-	Secondary RAT Type This field contains the RAT type for the secondary RAT.
-	Charging ID This field contains the Charging ID of the bearer corresponding to the reported usage. Only needed if IP-CAN session level charging is applied.
...
This retrieval is performed from the document 32298-i40.docx.




    Please provide the answers to the following multiple choice question.
    How can cloud computing benefit context management in the IoT (Internet of Things)?
    The output should be in the format: Option <Option id>

    Options:
    Write only the option number corresponding to the correct answer:\n
    option 1: By increasing interoperability among different IoT solutions
    option 2: By saving energy and optimizing actions
    option 3: By allowing real-time context stream processing
    option 4: By providing significant processing power and storage capabilities
    option 5: None of the above

    Output:
'''


# Prepare the data payload as a dictionary
data = {
    "inputs": question_184,
    "parameters": {
        "max_new_tokens": 50,
        "max_tokens": 2500,
        "return_full_text": False,
        "do_sample": True,
        "top_k": 10,
        # "stop": [
        #     "<|endoftext|>"
        # ],
    }
}

# Convert the dictionary to a JSON string
json_data = json.dumps(data)

# Set the appropriate headers for a JSON payload
headers = {
    "Content-Type": "application/json"
}

try:
    response = requests.post(url, data=json_data, headers=headers)
    print("Status Code:", response.status_code)
    print("Response Body:", response.text)
except Exception as e:
    print("An error occurred:", e)




# removed from context

    # Retrieval 6:
    # ...      type: object
    #       properties:
    #         singleNSSAI:
    #           $ref: 'TS29571_CommonData.yaml#/components/schemas/Snssai'
    #       required:
    #         - singleNSSAI
    #     NetworkSlicingInfo:
    #       type: object
    #       properties:
    #         sNSSAI:
    #           $ref: 'TS29571_CommonData.yaml#/components/schemas/Snssai'
    #         hPlmnSNSSAI:
    #           $ref: 'TS29571_CommonData.yaml#/components/schemas/Snssai'
    #       required:
    #         - sNSSAI
    #     PDUAddress:
    #       type: object
    #       properties:
    #         pduIPv4Address:
    #           ...
    # This retrieval is performed from the document 32291-i41.docx.


    # Retrieval 7:
    # ...shall not exceed 0.1% at the SNR given in table 8.3.3.1.2-1 and table 8.3.3.1.2-2.
    # Table 8.3.3.1.2-1: Minimum requirements for PUCCH format 1, 15 kHz SCS and 5MHz channel bandwidth

    # Table 8.3.3.1.2-2: Minimum requirements for PUCCH format 1, 30 kHz SCS and 10MHz channel bandwidth

    # 8.3.3.2	ACK missed detection requirements
    # 8.3.3.2.1	General
    # The ACK missed detection probability is the probability of not detecting an ACK when an ACK was sent. The test parameters in table 8.3.3.1.1-1 are configured.
    # ...
    # This retrieval is performed from the document 38108-i10.docx.


    # Retrieval 8:
    # ...The following capture the results for TRP muting in multi-TRP operation.
    # Table 6.3.2.2-1: BS energy savings by TRP muting in multi-TRP operation

    # For two TRP configuration case at different loads,
    # -	(2 sources) with semi-static TRP reduction, BS energy saving gain can be achieved by 36.9%~41.6% compared to no TRP reduction, with UPT loss of 7.27%~22%;
    # -	(one source) with dynamic TRP reduction, compared to no TRP reduction, BS energy saving gain can be achieved by 19.7%~28.7%, without reported UPT ...
    # This retrieval is performed from the document 38864-i10.docx.


    # Retrieval 9:
    # ...options
    # The test cases in this specification cover both NR/5GC (including FR1+FR2 CA or FR1+FR2 NR-DC) as well as EN-DC, NE-DC and NGEN-DC testing. Below shall be the understanding with respect to coverage across 5G NR connectivity options:
    # 1)	Unless otherwise stated within the test case, it shall be understood that test requirements are agnostic of the EN-DC, NE-DC and NGEN-DC connectivity options configured within the test. The test coverage across the EN-DC, NE-DC and NGEN-DC connectivity options ...
    # This retrieval is performed from the document 38521-3-i11.docx.


    # Retrieval 10:
    # ...and it is observed that SINR level is high and clearly sufficient to support high mobility performance except in the cases with DRX 40-160 ms and train traveling to opposite direction.

    # Figure 6.3.4.1.1.1-5 SINR distributions
    # 6.3.4.1.1.2	Uni-directional Scenario-A with DPS
    # This section shows system level simulation mobility performance results for uni-directional Scenario-A with DPS. Some of the observed statistics are of different type than in the section without DPS. Figure 6.3.4.1.1.2-1 shows ...
    # This retrieval is performed from the document 38854-i00.docx.