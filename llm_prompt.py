from openai import OpenAI

with open('chat-token.txt', 'r') as file:
    api_key = file.readline().strip()
client = OpenAI(api_key=api_key)


"""Fxns to generate prompts using Chat GPT API"""
def generate_basic_prompt(fid_report_main, missing_outcome, xgb_report=''):
    """
    Generate a prompt for GPT to predict a missing outcome for a specific Fontan patient based on MRI reports
    and surgical histories of similar patients.

    Parameters:
        fid_report_main (dict): Main patient's records including their CMR data, surgical data, and known outcomes.
        missing_outcome (list): list of status event (patient outcome) variables to occlude from patient report. Used in prompt to nudge

    Returns:
        str: A formatted prompt for GPT.
    """

    # Start constructing the prompt
    prompt = (
        "You are assisting with a clinical research project aimed at predicting specific outcomes for "
        "single-ventricle patients who have undergone Fontan surgery. The data for these patients is "
        "longitudinal and centered around MRI evaluations.\n\n"
        "**Objective:**\n"
        "For the given main patient, predict one of their clinical outcomes that has been intentionally "
        "occluded. You are provided with the following:\n"
        "1. The main patient's MRI report, surgical history, and known outcomes (excluding the occluded outcome).\n"
        "**Task:**\n"
        "Based on the provided data, predict the missing clinical outcome for the main patient and provide "
        "a reasoned explanation in 3-4 sentences. Ensure your reasoning considers:\n"
        "- Patterns or findings in the main patient's MRI report and surgical history.\n\n"
        f"{fid_report_main}\n"
    )

    prompt += (f"\n{xgb_report}\n")

    # Closing the prompt
    prompt += (
        f"\nUsing the above information, predict the missing clinical outcome, {missing_outcome} (yes or no) for the patient. "
        "Provide your reasoning in 3-4 sentences, considering both the  patient's data."
    )

    return prompt


def generate_prompt(fid_report_main, cmr_sub_reports, pthx_sub_reports, top_k_matches, missing_outcome, xgb_report=''):
    """
    Generate a prompt for GPT to predict a missing outcome for a specific Fontan patient based on MRI reports
    and surgical histories of similar patients.

    Parameters:
        fid_report_main (dict): Main patient's records including their CMR data, surgical data, and known outcomes.
        cmr_sub_reports (list of dict): List of CMR sub-reports for similar patients, including their outcome data.
        pthx_sub_reports (list of dict): List of surgical history sub-reports for similar patients, including their outcome data.
        top_k_matches (int): Number of similar patients used for context.

    Returns:
        str: A formatted prompt for GPT.
    """

    # Start constructing the prompt
    prompt = (
        "You are assisting with a clinical research project aimed at predicting specific outcomes for "
        "single-ventricle patients who have undergone Fontan surgery. The data for these patients is "
        "longitudinal and centered around MRI evaluations.\n\n"
        "**Objective:**\n"
        "For the given main patient, predict one of their clinical outcomes that has been intentionally "
        "occluded. You are provided with the following:\n"
        "1. The main patient's MRI report, surgical history, and known outcomes (excluding the occluded outcome).\n"
        "2. Data from {top_k_matches} similar patients, identified using an MRI-centered retrieval framework. "
        "This includes their MRI reports, surgical histories, and known outcomes, which can provide insights into the occluded outcome.\n\n"
        "**Task:**\n"
        "Based on the provided data, predict the missing clinical outcome for the main patient and provide "
        "a reasoned explanation in 3-4 sentences. Ensure your reasoning considers:\n"
        "- Patterns or findings in the main patient's MRI report and surgical history.\n"
        "- Trends or correlations observed in the data of the similar patients.\n\n"
        "**Main Patient:**\n"
        f"{fid_report_main}\n"
    )

    # Add similar patients' CMR sub-reports
    prompt += "**Similar Patients (CMR Reports):**\n"
    for i, cmr_report in enumerate(cmr_sub_reports[:top_k_matches], 1):
        prompt += f"{i}. {cmr_report}\n"

    # Add similar patients' surgical history sub-reports
    prompt += "\n**Similar Patients (Surgical History Reports):**\n"
    for i, pthx_report in enumerate(pthx_sub_reports[:top_k_matches], 1):
        prompt += f"{i}. {pthx_report}\n"

    prompt += (f"\n{xgb_report}\n")

    # Closing the prompt
    prompt += (
        f"\nUsing the above information, predict the missing clinical outcome, {missing_outcome} "
        f"(yes or no) for the main patient. "
        "Provide your reasoning in 3-4 sentences (don't just say yes or no!), considering both "
        "the main patient's data and the trends observed in the similar patients."
    )

    return prompt


def prompt_llm(fid_report_main, cmr_sub_reports, pthx_sub_reports, missing_outcome,
               top_k_matches=3,model_type='gpt-4o-mini', basic=False, xgb_report=''):
    if basic:
        prompt = generate_basic_prompt(fid_report_main, missing_outcome, xgb_report=xgb_report)
    else:
        prompt = generate_prompt(fid_report_main, cmr_sub_reports, pthx_sub_reports, top_k_matches, missing_outcome, xgb_report=xgb_report)

    chat_completion = client.chat.completions.create(
        model=model_type,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )

    resp = chat_completion.choices[0].to_dict()

    return resp['message']['content']