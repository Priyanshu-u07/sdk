# Copyright 2025 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model_registry.types import (
        ModelArtifact,
        ModelVersion,
        RegisteredModel,
        SupportedTypes,
    )


class ModelRegistryClient:
    """Client for Kubeflow Model Registry operations.

    Requires the model-registry package to be installed. Install it with:

        pip install 'kubeflow[hub]'

    """

    def __init__(
        self,
        base_url: str,
        port: int | None = None,
        *,
        author: str | None = None,
        is_secure: bool | None = None,
        user_token: str | None = None,
        custom_ca: str | None = None,
    ):
        """Initialize the ModelRegistryClient.

        Args:
            base_url: Base URL of the model registry server including scheme.
                Examples: "https://registry.example.com", "http://localhost".

        Keyword Args:
            port: Server port. If not provided, inferred from `base_url` scheme:
                - https:// defaults to 443
                - http:// defaults to 8080
                - no scheme defaults to 443
            author: Name of the author.
            is_secure: Whether to use a secure connection. If not provided, 
                inferred from `base_url`:
                - https:// sets is_secure=True
                - http:// sets is_secure=False
                - no scheme defaults to True
            user_token: The PEM-encoded user token as a string.
            custom_ca: Path to the PEM-encoded root certificates as a string.

        Raises:
            ImportError: If the `model-registry` package is not installed.

        Example:
            from kubeflow.hub import ModelRegistryClient

            client = ModelRegistryClient(base_url="http://localhost:8080")
        """
        try:
            from model_registry import ModelRegistry
        except ImportError as e:
            raise ImportError(
                "model-registry is not installed. Install it with:\n\n"  # fmt: skip
                "  pip install 'kubeflow[hub]'\n"
            ) from e

        is_http = base_url.startswith("http://")
        if is_secure is None:
            is_secure = not is_http
        if port is None:
            port = 8080 if is_http else 443

        self._registry = ModelRegistry(
            server_address=base_url,
            port=port,
            author=author,  # type: ignore[arg-type]
            is_secure=is_secure,
            user_token=user_token,
            custom_ca=custom_ca,
        )

    def register_model(
        self,
        name: str,
        uri: str,
        *,
        version: str,
        model_format_name: str | None = None,
        model_format_version: str | None = None,
        author: str | None = None,
        owner: str | None = None,
        version_description: str | None = None,
        metadata: Mapping[str, SupportedTypes] | None = None,
    ) -> RegisteredModel:
        """Register a new model or a new version of an existing model.

        The model must be stored in external storage (e.g., S3, GCS) before 
        registration. The URI should point to the model artifacts.

        Args:
            name: The name of the registered model.
            uri: The URI where the model artifacts are stored.

        Keyword Args:
            version: The version string for this registration. Must be unique 
                for this model name.
            model_format_name: The format of the model (e.g., "pytorch", 
                "tensorflow"). Used by KServe for inference.
            model_format_version: The version of the model format (e.g., "2.0").
            author: The author of the model. Defaults to the client author.
            owner: The owner of the model. Defaults to the client author.
            version_description: A description of this specific model version.
            metadata: A dictionary of additional metadata to store with the version.

        Returns:
            model_registry.types.RegisteredModel: The registered model object.

        Raises:
            model_registry.exceptions.StoreError: If the registry backend fails 
                to register the model.

        Example:
            from kubeflow.hub import ModelRegistryClient

            client = ModelRegistryClient(base_url="http://localhost:8080")

            model = client.register_model(
                name="mnist-classifier",
                uri="s3://my-bucket/models/mnist/v1/",
                version="v1.0.0",
                model_format_name="pytorch",
                version_description="Initial release of MNIST model"
            )
            print(f"Registered model ID: {model.id}")
        """
        return self._registry.register_model(
            name=name,
            uri=uri,
            model_format_name=model_format_name,  # type: ignore[arg-type]
            model_format_version=model_format_version,  # type: ignore[arg-type]
            version=version,
            author=author,
            owner=owner,
            description=version_description,
            metadata=metadata,
        )

    def update_model(self, model: RegisteredModel) -> RegisteredModel:
        """Update the metadata of an existing registered model.

        Args:
            model: The `RegisteredModel` instance to update. It must have 
                a valid ID.

        Returns:
            model_registry.types.RegisteredModel: The updated registered model.

        Raises:
            TypeError: If the input is not a `RegisteredModel` instance.
            model_registry.exceptions.StoreError: If the registered model does 
                not have an ID.

        Example:
            from kubeflow.hub import ModelRegistryClient

            client = ModelRegistryClient(base_url="http://localhost:8080")
            model = client.get_model(name="mnist-classifier")
            
            # Update description
            model.description = "Updated description"
            updated_model = client.update_model(model)
        """
        from model_registry.types import RegisteredModel

        if not isinstance(model, RegisteredModel):
            raise TypeError(f"Expected RegisteredModel, got {type(model).__name__}. ")
        return self._registry.update(model)

    def update_model_version(self, model_version: ModelVersion) -> ModelVersion:
        """Update an existing model version's metadata.

        Args:
            model_version: The `ModelVersion` instance to update. It must have 
                a valid ID.

        Returns:
            model_registry.types.ModelVersion: The updated model version.

        Raises:
            TypeError: If the input is not a `ModelVersion` instance.
            model_registry.exceptions.StoreError: If the version does not have an ID.

        Example:
            from kubeflow.hub import ModelRegistryClient

            client = ModelRegistryClient(base_url="http://localhost:8080")
            version = client.get_model_version(name="mnist", version="v1.0.0")
            
            # Update metadata
            version.metadata["accuracy"] = 0.98
            client.update_model_version(version)
        """
        from model_registry.types import ModelVersion

        if not isinstance(model_version, ModelVersion):
            raise TypeError(f"Expected ModelVersion, got {type(model_version).__name__}. ")
        return self._registry.update(model_version)

    def update_model_artifact(self, model_artifact: ModelArtifact) -> ModelArtifact:
        """Update an existing model artifact's metadata.

        Args:
            model_artifact: The `ModelArtifact` instance to update. It must 
                have a valid ID.

        Returns:
            model_registry.types.ModelArtifact: The updated model artifact.

        Raises:
            TypeError: If the input is not a `ModelArtifact` instance.
            model_registry.exceptions.StoreError: If the artifact does not have an ID.

        Example:
            from kubeflow.hub import ModelRegistryClient

            client = ModelRegistryClient(base_url="http://localhost:8080")
            artifact = client.get_model_artifact(name="mnist", version="v1.0.0")
            
            # Update artifact description
            artifact.description = "Production-ready weights"
            client.update_model_artifact(artifact)
        """
        from model_registry.types import ModelArtifact

        if not isinstance(model_artifact, ModelArtifact):
            raise TypeError(f"Expected ModelArtifact, got {type(model_artifact).__name__}. ")
        return self._registry.update(model_artifact)

    def get_model(self, name: str) -> RegisteredModel:
        """Get a specific registered model by name.

        Args:
            name: The name of the registered model.

        Returns:
            model_registry.types.RegisteredModel: The registered model object.

        Raises:
            ValueError: If a registered model with the given `name` is not found.

        Example:
            from kubeflow.hub import ModelRegistryClient

            client = ModelRegistryClient(base_url="http://localhost:8080")
            model = client.get_model(name="mnist-classifier")
            print(f"Model ID: {model.id}")
        """
        model = self._registry.get_registered_model(name)
        if model is None:
            raise ValueError(f"Model {name!r} not found")
        return model

    def get_model_version(self, name: str, version: str) -> ModelVersion:
        """Get a specific model version.

        Args:
            name: The name of the registered model.
            version: The version string to retrieve.

        Returns:
            model_registry.types.ModelVersion: The model version object.

        Raises:
            model_registry.exceptions.StoreError: If the registered model does 
                not exist.
            ValueError: If the version string is not found for the given 
                registered model.

        Example:
            from kubeflow.hub import ModelRegistryClient

            client = ModelRegistryClient(base_url="http://localhost:8080")
            version = client.get_model_version(name="mnist", version="v1.0.0")
            print(f"Version ID: {version.id}")
        """
        model_version = self._registry.get_model_version(name, version)
        if model_version is None:
            raise ValueError(f"Model version {version!r} not found for model {name!r}")
        return model_version

    def get_model_artifact(self, name: str, version: str) -> ModelArtifact:
        """Get the artifact associated with a specific model version.

        Args:
            name: The name of the registered model.
            version: The version of the registered model.

        Returns:
            model_registry.types.ModelArtifact: The model artifact object.

        Raises:
            model_registry.exceptions.StoreError: If either the registered 
                model or version does not exist.
            ValueError: If the artifact is not found.

        Example:
            from kubeflow.hub import ModelRegistryClient

            client = ModelRegistryClient(base_url="http://localhost:8080")
            artifact = client.get_model_artifact(name="mnist", version="v1.0.0")
            print(f"Artifact URI: {artifact.uri}")
        """
        artifact = self._registry.get_model_artifact(name, version)
        if artifact is None:
            raise ValueError(f"Model artifact not found for model {name!r} version {version!r}")
        return artifact

    def list_models(self) -> Iterator[RegisteredModel]:
        """Get an iterator for all registered models.

        Yields:
            model_registry.types.RegisteredModel: The next registered model.

        Example:
            from kubeflow.hub import ModelRegistryClient

            client = ModelRegistryClient(base_url="http://localhost:8080")
            for model in client.list_models():
                print(f"Model: {model.name}")
        """
        yield from self._registry.get_registered_models()

    def list_model_versions(self, name: str) -> Iterator[ModelVersion]:
        """Get an iterator for all versions of a specific registered model.

        Args:
            name: The name of the registered model.

        Yields:
            model_registry.types.ModelVersion: The next model version.

        Raises:
            model_registry.exceptions.StoreError: If the registered model does 
                not exist.

        Example:
            from kubeflow.hub import ModelRegistryClient

            client = ModelRegistryClient(base_url="http://localhost:8080")
            for version in client.list_model_versions(name="mnist"):
                print(f"Version: {version.version}")
        """
        yield from self._registry.get_model_versions(name)
