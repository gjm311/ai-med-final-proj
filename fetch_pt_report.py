"""Fxns for creating pt specific reports"""
def get_fid_dict(fid, tr_tst_dict):
    fid_dict = {key: {fid: value[fid]} for key, value in tr_tst_dict.items() if fid in value.keys()}
    return fid_dict


def generate_patient_report(patient_data, psu_occlusions):
    report_sections = []
    psu_report_sections = []
    for category, instances in patient_data.items():
        # if category in ['cmr', 'pt_clinical_surgical_history', 'pt_status_event_update']:
        if category in ['pt_status_event_update']:
            section = f" {category.replace('_', ' ').title()} \n"
            psu_section = f""
            for patient_id, events in instances.items():
                for event_id, event_data in events.items():
                    if event_id not in [0, 1]:
                        continue
                    # section += f"\n  Instance {event_id}:\n"
                    for data_type, details in event_data.items():
                        if data_type != 'metadata':
                            for key, value in details.items():
                                if str(value) == 'nan' or 'month' in key.lower() or 'year' in key.lower() or 'mnth' in key.lower() or 'Other relevant comments or concerns' in key:
                                    continue
                                if key not in psu_occlusions:
                                    # if not in list to occlude, storing data in report, otherwise only storing in psu report (used to evaluate resp)
                                    section += f" {key.replace('Does the patient have a', '')} {value}\n"
                                    # psu_section += f" {key} {value}\n"
                                else:
                                    psu_section += f" {key.replace('Does the patient have a', '')} {value}\n"
            if category == 'pt_status_event_update':
                psu_report_sections.append(psu_section)
            report_sections.append(section)

    return "\n".join(report_sections), psu_report_sections


def generate_patient_subreport(patient_data, form_nm, cdi, distance, psu_occlusions=[]):
    if len(psu_occlusions) == 0:
        section = f" {form_nm.replace('_', ' ').title()} for similar patient #{cdi + 1} with cos similarity measure: {distance}\n"
    else:
        section = f""

    for data_type, details in patient_data.items():
        if data_type != 'metadata':
            for key, value in details.items():
                if str(value) == 'nan' or 'month' in key.lower() or 'year' in key.lower() or 'mnth' in key.lower():
                    continue

                if len(psu_occlusions) == 0:
                    section += f" {key.replace('Does the patient have a', '')} {value}\n"
                elif key in psu_occlusions:
                    section += f" {key.replace('Does the patient have a', '')} {value}\n"

    return section


def fetch_sub_reports(form_nm, faiss_results, report_data_dict, psu_occlusions):
    reports = []
    psu_reports = []
    report_sections = []

    # fetching reports for given
    for cdi, close_dict in enumerate(faiss_results):
        # getting force id, instance number closest to cmr in question and generating sub report
        fid_of_interest = close_dict['force_id']
        inst_of_interest = close_dict['instance']
        distance = close_dict['distance']
        sub_fid_dict = get_fid_dict(fid_of_interest, report_data_dict)

        # getting data relevant to form/fid/instance #
        sub_fid_dict_curr = sub_fid_dict[form_nm][fid_of_interest][inst_of_interest]
        sub_fid_report = generate_patient_subreport(sub_fid_dict_curr, form_nm=form_nm, cdi=cdi, distance=distance)

        # getting data relvant to pt status upd for testing
        sub_psu_fid_dict = sub_fid_dict['pt_status_event_update'][fid_of_interest][0]
        sub_psu_report = generate_patient_subreport(sub_psu_fid_dict, form_nm='pt_status_event_update', cdi=cdi,
                                                    distance=distance)

        # aggregating results; return psu/psu + other separately so we feed the reports to gpt and have something to look at during testing....
        report_sections.append(sub_fid_report)
        report_sections.append(sub_psu_report)
        reports.append("\n".join(report_sections))

        # getting occlusions only
        sub_occl_psu_report = generate_patient_subreport(sub_psu_fid_dict, form_nm='pt_status_event_update', cdi=cdi,
                                                         distance=distance, psu_occlusions=psu_occlusions)
        psu_reports.append(sub_occl_psu_report)

    return reports, psu_reports