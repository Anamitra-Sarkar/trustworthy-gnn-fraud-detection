from pathlib import Path
import json
import os
import shutil
import traceback

from kaggle.api import kaggle_api_extended as ext
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kaggle_object import FieldMetadata
from slugify import slugify

ROOT = Path(__file__).resolve().parent


def _patch_kernel_request_model():
    """Inject the machine_shape field into the generated request model."""
    req_cls = ext.ApiSaveKernelRequest
    if not any(field.field_name == "machine_shape" for field in req_cls._fields):
        string_serializer = next(
            field.serializer for field in req_cls._fields if field.field_name == "language"
        )
        req_cls._fields.append(
            FieldMetadata(
                json_name="machineShape",
                field_name="machine_shape",
                private_field_name="_machine_shape",
                field_type=str,
                default_value=None,
                serializer=string_serializer,
                optional=True,
            )
        )


def kernels_push_with_machine_shape(api, folder, timeout=None):
    """Copy of KaggleApi.kernels_push with machine_shape support."""
    if not os.path.isdir(folder):
        raise ValueError("Invalid folder: " + folder)

    meta_file = os.path.join(folder, api.KERNEL_METADATA_FILE)
    if not os.path.isfile(meta_file):
        raise ValueError("Metadata file not found: " + str(meta_file))

    with open(meta_file) as f:
        meta_data = json.load(f)

    title = api.get_or_default(meta_data, "title", None)
    if title and len(title) < 5:
        raise ValueError("Title must be at least five characters")

    code_path = api.get_or_default(meta_data, "code_file", "")
    if not code_path:
        raise ValueError("A source file must be specified in the metadata")

    code_file = os.path.join(folder, code_path)
    if not os.path.isfile(code_file):
        raise ValueError("Source file not found: " + str(code_file))

    slug = meta_data.get("id")
    id_no = meta_data.get("id_no")
    if not slug and not id_no:
        raise ValueError("ID or slug must be specified in the metadata")
    if slug:
        api.validate_kernel_string(slug)
        if "/" in slug:
            kernel_slug = slug.split("/")[1]
        else:
            kernel_slug = slug
        if title:
            as_slug = slugify(title)
            if kernel_slug.lower() != as_slug:
                print(
                    "Your kernel title does not resolve to the specified "
                    "id. This may result in surprising behavior. We "
                    "suggest making your title something that resolves to "
                    "the specified id. See %s for more information on "
                    "how slugs are determined."
                    % "https://en.wikipedia.org/wiki/Clean_URL#Slug"
                )

    language = api.get_or_default(meta_data, "language", "")
    if language not in api.valid_push_language_types:
        raise ValueError(
            "A valid language must be specified in the metadata. Valid "
            "options are " + str(api.valid_push_language_types)
        )

    kernel_type = api.get_or_default(meta_data, "kernel_type", "")
    if kernel_type not in api.valid_push_kernel_types:
        raise ValueError(
            "A valid kernel type must be specified in the metadata. Valid "
            "options are " + str(api.valid_push_kernel_types)
        )

    if kernel_type == "notebook" and language == "rmarkdown":
        language = "r"

    dataset_sources = api.get_or_default(meta_data, "dataset_sources", [])
    for source in dataset_sources:
        api.validate_dataset_string(source)

    kernel_sources = api.get_or_default(meta_data, "kernel_sources", [])
    for source in kernel_sources:
        api.validate_kernel_string(source)

    model_sources = api.get_or_default(meta_data, "model_sources", [])
    for source in model_sources:
        api.validate_model_string(source)

    docker_pinning_type = api.get_or_default(
        meta_data, "docker_image_pinning_type", None
    )
    if (
        docker_pinning_type is not None
        and docker_pinning_type not in api.valid_push_pinning_types
    ):
        raise ValueError(
            "If specified, the docker_image_pinning_type must be one of "
            + str(api.valid_push_pinning_types)
        )

    with open(code_file) as f:
        script_body = f.read()

    if kernel_type == "notebook":
        json_body = json.loads(script_body)
        if "cells" in json_body:
            for cell in json_body["cells"]:
                if "outputs" in cell and cell["cell_type"] == "code":
                    cell["outputs"] = []
                if "source" in cell and isinstance(cell["source"], list):
                    cell["source"] = "".join(cell["source"])
        script_body = json.dumps(json_body)

    with api.build_kaggle_client() as kaggle:
        request = ext.ApiSaveKernelRequest()
        request.id = id_no
        request.slug = slug
        request.new_title = api.get_or_default(meta_data, "title", None)
        request.text = script_body
        request.language = language
        request.kernel_type = kernel_type
        request.is_private = api.get_bool(meta_data, "is_private", True)
        request.enable_gpu = api.get_bool(meta_data, "enable_gpu", False)
        request.enable_tpu = api.get_bool(meta_data, "enable_tpu", False)
        request.enable_internet = api.get_bool(
            meta_data, "enable_internet", True
        )
        request.dataset_data_sources = dataset_sources
        request.competition_data_sources = api.get_or_default(
            meta_data, "competition_sources", []
        )
        request.kernel_data_sources = kernel_sources
        request.model_data_sources = model_sources
        request.category_ids = api.get_or_default(meta_data, "keywords", [])
        request.docker_image_pinning_type = docker_pinning_type
        machine_shape = api.get_or_default(meta_data, "machine_shape", None)
        if machine_shape:
            object.__setattr__(request, "_machine_shape", machine_shape)
        if timeout:
            request.session_timeout_seconds = int(timeout)
        return kaggle.kernels.kernels_api_client.save_kernel(request)


def main():
    token_path = Path("/home/anamitra/Downloads/API_Keys_and_Secrets/hf_token")
    if not token_path.exists():
        print("HF token not found at", token_path)
        return

    token = token_path.read_text().strip()
    _patch_kernel_request_model()

    api = KaggleApi()
    api.authenticate()

    scripts = [
        ("kaggle_train_gnn.py", "gnn-kernel-metadata.json"),
        ("kaggle_train_edl.py", "edl-kernel-metadata.json"),
    ]

    for script_name, metadata_name in scripts:
        print(f"Preparing to push {script_name} using metadata {metadata_name}...")

        script_path = ROOT / script_name
        metadata_path = ROOT / metadata_name
        kernel_metadata = ROOT / "kernel-metadata.json"
        backup_name = script_path.with_suffix(script_path.suffix + ".bak")
        meta_backup = ROOT / "kernel-metadata.json.bak"

        shutil.copy(script_path, backup_name)
        if kernel_metadata.exists():
            shutil.copy(kernel_metadata, meta_backup)

        try:
            content = script_path.read_text()
            script_path.write_text(content.replace("PLACEHOLDER_HF_TOKEN", token))
            shutil.copy(metadata_path, kernel_metadata)

            print(f"Pushing {script_name} to Kaggle with machine_shape metadata...")
            res = kernels_push_with_machine_shape(api, str(ROOT))
            print("Result:", res)
        except Exception:
            traceback.print_exc()
        finally:
            if backup_name.exists():
                shutil.move(backup_name, script_path)
            if meta_backup.exists():
                shutil.move(meta_backup, kernel_metadata)
            elif kernel_metadata.exists():
                kernel_metadata.unlink()

        print(f"Finished pushing {script_name}.\n")


if __name__ == "__main__":
    main()
