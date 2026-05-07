import os
import shutil
from typing import cast

import streamlit as st

from orchestrator import AgentState, OrchestratorExecutionError, factory


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_FILENAME = "reference.jpg"
JOB_TYPES = ("NEW", "REDESIGN")


def extract_live_preview_url(debug_logs: str) -> str | None:
    marker = "SUCCESS! Live Preview URL:"
    for line in debug_logs.splitlines():
        if marker in line:
            return line.split(marker, 1)[1].strip()
    return None


def normalize_project_name(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "-").replace("_", "-")
    cleaned = []
    previous_was_dash = False

    for char in normalized:
        if char.isalnum():
            cleaned.append(char)
            previous_was_dash = False
        elif char == "-":
            if not previous_was_dash:
                cleaned.append(char)
            previous_was_dash = True

    return "".join(cleaned).strip("-")


def save_uploaded_file(uploaded_file, destination_path: str) -> None:
    uploaded_file.seek(0)
    with open(destination_path, "wb") as destination:
        shutil.copyfileobj(uploaded_file, destination)


def render_failure(exc: OrchestratorExecutionError) -> None:
    st.error(f"Website pipeline failed: {exc}")

    debug_logs = exc.state.get("debug_logs", "")
    error_traceback = exc.state.get("error_traceback", "")

    if debug_logs:
        with st.expander("Debug Logs"):
            st.code(debug_logs, language="text")

    if error_traceback:
        with st.expander("Error Traceback"):
            st.code(error_traceback, language="text")


def render_success(final_state: AgentState, message: str) -> None:
    st.success(message)

    site_manifest = final_state.get("site_manifest", "")
    if site_manifest:
        st.code(site_manifest, language="text")

    build_errors = final_state.get("build_errors", "")
    build_logs = final_state.get("build_logs", "")
    debug_logs = final_state.get("debug_logs", "")
    live_preview_url = extract_live_preview_url(debug_logs)

    if live_preview_url:
        st.link_button("Open Live Preview", live_preview_url, use_container_width=True)

    if build_errors:
        with st.expander("Build Errors"):
            st.code(build_errors, language="text")

    if build_logs:
        with st.expander("Build Logs"):
            st.code(build_logs, language="text")

    if debug_logs:
        with st.expander("Debug Logs"):
            st.code(debug_logs, language="text")


def invoke_factory(state: AgentState) -> AgentState:
    return cast(AgentState, factory.invoke(state))


def base_state(project_name: str, business_name: str) -> AgentState:
    return {
        "project_id": project_name,
        "business_name": business_name,
        "job_type": "",
        "requirements": "",
        "assets": [],
        "edit_mode": False,
        "edit_prompt": "",
        "theme": {},
        "site_meta": {},
        "site_manifest": "",
        "build_logs": "",
        "build_errors": "",
        "iteration_count": 0,
        "current_stage": "",
        "debug_logs": "",
        "error_message": "",
        "error_traceback": "",
        "provider_response_debug": "",
    }


st.set_page_config(
    page_title="Site Factory | Agency Dashboard",
    layout="wide",
)

st.title("Site Factory | Agency Dashboard")
st.caption("Launch a new website or edit an existing generated site.")

mode = st.radio("Mode", ["Generate", "Edit"], horizontal=True)

if mode == "Generate":
    with st.container(border=True):
        left_col, right_col = st.columns((3, 2), gap="large")

        with left_col:
            st.subheader("Project Brief")
            business_name = st.text_input(
                "Business Name",
                placeholder="Apex Dental Care",
                help="This is the public-facing brand name used in the generated site shell.",
                key="generate_business_name",
            )
            project_name_input = st.text_input(
                "Project Name",
                placeholder="smile-clinic-01",
                help="Lowercase only. Spaces will be converted to hyphens automatically.",
                key="generate_project_name",
            )
            normalized_project_name = normalize_project_name(project_name_input)

            if project_name_input and normalized_project_name != project_name_input.strip():
                st.info(f"Project ID will be used as `{normalized_project_name}`.")

            job_type = st.selectbox("Job Type", JOB_TYPES, index=0, key="generate_job_type")
            business_description = st.text_area(
                "Business Description",
                placeholder=(
                    "Describe the business, audience, goals, required pages, tone, and any "
                    "must-have messaging for the generated site."
                ),
                height=220,
                key="generate_business_description",
            )
            theme_overrides = st.text_input(
                "Theme Overrides",
                placeholder="Example: Use warm neutrals, premium editorial typography, muted gold accents.",
                key="generate_theme_overrides",
            )

        with right_col:
            st.subheader("Uploads")
            reference_image = st.file_uploader(
                "Reference Image",
                type=["png", "jpg", "jpeg"],
                help="Required. This image is saved as `reference.jpg` for the pipeline.",
                key="generate_reference_image",
            )
            asset_files = st.file_uploader(
                "Assets",
                accept_multiple_files=True,
                help="Optional. Upload logos, product images, videos, PDFs, or any supporting files.",
                key="generate_assets",
            )

            with st.container(border=True):
                st.markdown("**Submission Checklist**")
                st.write("- Provide a valid project name.")
                st.write("- Add a business description with enough detail to guide the build.")
                st.write("- Upload a reference image.")

    validation_errors = []
    if not business_name.strip():
        validation_errors.append("Enter a business name.")
    if not normalized_project_name:
        validation_errors.append("Enter a project name.")
    if not business_description.strip():
        validation_errors.append("Add a business description.")
    if reference_image is None:
        validation_errors.append("Upload a reference image.")

    if validation_errors:
        st.warning("Complete the required fields before generating the website.")
        for error in validation_errors:
            st.caption(f"- {error}")

    generate_clicked = st.button(
        "Generate Website",
        type="primary",
        use_container_width=True,
        disabled=bool(validation_errors),
        key="generate_button",
    )

    if generate_clicked:
        requirements = business_description.strip()
        if theme_overrides.strip():
            requirements = f"{requirements}\n\nTheme Overrides:\n{theme_overrides.strip()}"

        with st.spinner("Building factory pipeline..."):
            try:
                reference_path = os.path.join(ROOT_DIR, REFERENCE_FILENAME)
                save_uploaded_file(reference_image, reference_path)

                asset_filenames = []
                for asset_file in asset_files or []:
                    asset_path = os.path.join(ROOT_DIR, asset_file.name)
                    save_uploaded_file(asset_file, asset_path)
                    asset_filenames.append(asset_file.name)

                asset_filenames.append(REFERENCE_FILENAME)

                state = base_state(normalized_project_name, business_name.strip())
                state.update(
                    {
                        "job_type": job_type,
                        "requirements": requirements,
                        "assets": asset_filenames,
                        "edit_mode": False,
                        "edit_prompt": "",
                    }
                )

                final_state = invoke_factory(state)
            except OrchestratorExecutionError as exc:
                render_failure(exc)
            except Exception as exc:
                st.error(f"Website generation failed: {exc}")
            else:
                render_success(final_state, "Website generated successfully.")

else:
    with st.container(border=True):
        st.subheader("Edit Existing Site")
        edit_project_name_input = st.text_input(
            "Project ID",
            placeholder="smile-clinic-01",
            help="Must match the GitHub repository/project id for the existing generated site.",
            key="edit_project_name",
        )
        normalized_edit_project_name = normalize_project_name(edit_project_name_input)

        if edit_project_name_input and normalized_edit_project_name != edit_project_name_input.strip():
            st.info(f"Project ID will be used as `{normalized_edit_project_name}`.")

        edit_prompt = st.text_area(
            "Edit Prompt",
            placeholder="Example: Make the hero section more premium and add a financing CTA to the contact page.",
            height=220,
            key="edit_prompt",
        )
        edit_reference_image = st.file_uploader(
            "New Reference Image",
            type=["png", "jpg", "jpeg"],
            key="edit_reference_image",
        )
        edit_asset_files = st.file_uploader(
            "New Assets",
            accept_multiple_files=True,
            key="edit_assets",
        )

    edit_validation_errors = []
    if not normalized_edit_project_name:
        edit_validation_errors.append("Enter a project ID.")
    if not edit_prompt.strip():
        edit_validation_errors.append("Describe the requested edit.")

    if edit_validation_errors:
        st.warning("Complete the required fields before editing the website.")
        for error in edit_validation_errors:
            st.caption(f"- {error}")

    edit_clicked = st.button(
        "Edit Existing Site",
        type="primary",
        use_container_width=True,
        disabled=bool(edit_validation_errors),
        key="edit_button",
    )

    if edit_clicked:
        with st.spinner("Editing existing site..."):
            try:
                edit_assets_list: list[str] = []
                if edit_reference_image is not None:
                    edit_reference_filename = "edit_reference.jpg"
                    edit_reference_path = os.path.join(ROOT_DIR, edit_reference_filename)
                    save_uploaded_file(edit_reference_image, edit_reference_path)
                    edit_assets_list.append(edit_reference_filename)

                for asset_file in edit_asset_files or []:
                    asset_path = os.path.join(ROOT_DIR, asset_file.name)
                    save_uploaded_file(asset_file, asset_path)
                    edit_assets_list.append(asset_file.name)

                state = base_state(normalized_edit_project_name, "")
                state.update(
                    {
                        "job_type": "EDIT",
                        "assets": edit_assets_list,
                        "edit_mode": True,
                        "edit_prompt": edit_prompt.strip(),
                    }
                )

                final_state = invoke_factory(state)
            except OrchestratorExecutionError as exc:
                render_failure(exc)
            except Exception as exc:
                st.error(f"Website edit failed: {exc}")
            else:
                render_success(final_state, "Website edited successfully.")
