# Copyright 2024 The Kubeflow Authors.
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

from collections.abc import Callable, Iterator
import logging

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.trainer.backends.container.backend import ContainerBackend
from kubeflow.trainer.backends.container.types import ContainerBackendConfig
from kubeflow.trainer.backends.kubernetes.backend import KubernetesBackend
from kubeflow.trainer.backends.localprocess.backend import (
    LocalProcessBackend,
    LocalProcessBackendConfig,
)
from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types

logger = logging.getLogger(__name__)


class TrainerClient:
    def __init__(
        self,
        backend_config: KubernetesBackendConfig
        | LocalProcessBackendConfig
        | ContainerBackendConfig
        | None = None,
    ):
        """Initialize a Kubeflow Trainer client.

        Args:
            backend_config: Backend configuration. Either KubernetesBackendConfig,
                LocalProcessBackendConfig, ContainerBackendConfig, or None to use 
                the backend's default config class. Defaults to 
                KubernetesBackendConfig.

        Raises:
            ValueError: If the provided `backend_config` is invalid.

        Example:
            Initialize with the default Kubernetes backend:

            from kubeflow.trainer import TrainerClient

            client = TrainerClient()

            Initialize with a local process backend:

            from kubeflow.trainer import TrainerClient, LocalProcessBackendConfig

            client = TrainerClient(backend_config=LocalProcessBackendConfig())
        """
        # Set the default backend config.
        if not backend_config:
            backend_config = KubernetesBackendConfig()

        if isinstance(backend_config, KubernetesBackendConfig):
            self.backend = KubernetesBackend(backend_config)
        elif isinstance(backend_config, LocalProcessBackendConfig):
            self.backend = LocalProcessBackend(backend_config)
        elif isinstance(backend_config, ContainerBackendConfig):
            self.backend = ContainerBackend(backend_config)
        else:
            raise ValueError(f"Invalid backend config '{backend_config}'")

    def list_runtimes(self) -> list[types.Runtime]:
        """List the available Training Runtimes.

        Returns:
            list[kubeflow.trainer.types.Runtime]: A list of available training 
                runtimes. If no runtimes exist, an empty list is returned.

        Raises:
            TimeoutError: If the request to list runtimes times out.
            RuntimeError: If the client fails to list runtimes.

        Example:
            from kubeflow.trainer import TrainerClient

            client = TrainerClient()
            runtimes = client.list_runtimes()

            for runtime in runtimes:
                print(f"Runtime: {runtime.name}")
        """
        return self.backend.list_runtimes()

    def get_runtime(self, name: str) -> types.Runtime:
        """Get information about a specific Training Runtime.

        Args:
            name: The name of the Training Runtime to retrieve.

        Returns:
            kubeflow.trainer.types.Runtime: The Training Runtime object.

        Raises:
            ValueError: If the runtime `name` is invalid or not found.
            TimeoutError: If checking for the Training Runtime times out.
            RuntimeError: If the backend fails to retrieve the Training Runtime.

        Example:
            from kubeflow.trainer import TrainerClient

            client = TrainerClient()
            runtime = client.get_runtime(name="pytorch-distributed")
        """
        return self.backend.get_runtime(name=name)

    def get_runtime_packages(self, runtime: types.Runtime):
        """Print the installed Python packages for the given Training Runtime. 
        
        If the Training Runtime supports GPUs, it also prints available GPU 
        information for the training node.

        Args:
            runtime: A reference to an existing Training Runtime object, 
                typically obtained via `get_runtime` or `list_runtimes`.

        Raises:
            ValueError: If the input arguments are invalid.
            RuntimeError: If the client fails to get package information for 
                the Training Runtime.

        Example:
            from kubeflow.trainer import TrainerClient

            client = TrainerClient()
            runtime = client.get_runtime(name="pytorch-distributed")
            client.get_runtime_packages(runtime=runtime)
        """
        return self.backend.get_runtime_packages(runtime=runtime)

    def train(
        self,
        runtime: str | types.Runtime | None = None,
        initializer: types.Initializer | None = None,
        trainer: types.CustomTrainer
        | types.CustomTrainerContainer
        | types.BuiltinTrainer
        | None = None,
        options: list | None = None,
    ) -> str:
        """Create and start a TrainJob. 
        
        You can configure the TrainJob using one of these trainer types:
        - `CustomTrainer`: Runs training with a user-defined function.
        - `CustomTrainerContainer`: Runs training with a user-defined container image.
        - `BuiltinTrainer`: Uses a predefined trainer with built-in logic.

        Args:
            runtime: Reference to an existing Training Runtime. Accepts a 
                runtime name (str) or a Runtime object. Defaults to the 
                "torch-distributed" runtime if not provided.
            initializer: Optional configuration for dataset and model 
                initializers.
            trainer: Configuration for the trainer. If not specified, the 
                TrainJob uses the Training Runtime's default values.
            options: Optional list of configuration options to apply to the 
                TrainJob. Import options from `kubeflow.trainer.options`.

        Returns:
            str: The unique name of the generated TrainJob.

        Raises:
            ValueError: If the input arguments are invalid.
            TimeoutError: If the request to create the TrainJob times out.
            RuntimeError: If the backend fails to create the TrainJob.

        Example:
            from kubeflow.trainer import TrainerClient, types

            def train_func(parameters):
                # Your training logic here
                print(f"Training with parameters: {parameters}")

            client = TrainerClient()
            job_name = client.train(
                runtime="torch-distributed",
                trainer=types.CustomTrainer(func=train_func)
            )
            print(f"Started job: {job_name}")
        """
        return self.backend.train(
            runtime=runtime,
            initializer=initializer,
            trainer=trainer,
            options=options,
        )

    def list_jobs(self, runtime: Optional[types.Runtime] = None) -> list[types.TrainJob]:
        """List the generated TrainJobs. 
        
        If a Training Runtime is specified, only TrainJobs associated with 
        that runtime are returned.

        Args:
            runtime: Optional reference to an existing Training Runtime object 
                to filter TrainJobs.

        Returns:
            list[kubeflow.trainer.types.TrainJob]: A list of generated 
                TrainJob objects. If no jobs exist, an empty list is returned.

        Raises:
            TimeoutError: If the request to list TrainJobs times out.
            RuntimeError: If the backend fails to list TrainJobs.

        Example:
            from kubeflow.trainer import TrainerClient

            client = TrainerClient()
            jobs = client.list_jobs()
            for job in jobs:
                print(f"Job Name: {job.name}, Status: {job.status}")
        """
        return self.backend.list_jobs(runtime=runtime)

    def get_job(self, name: str) -> types.TrainJob:
        """Get information about a specific TrainJob.

        Args:
            name: The unique name of the TrainJob.

        Returns:
            kubeflow.trainer.types.TrainJob: The TrainJob object.

        Raises:
            TimeoutError: If the request to retrieve the job times out.
            RuntimeError: If the backend fails to retrieve the job.

        Example:
            from kubeflow.trainer import TrainerClient

            client = TrainerClient()
            job = client.get_job(name="my-training-job-abc123")
            print(f"Job Status: {job.status}")
        """

        return self.backend.get_job(name=name)

    def get_job_logs(
        self,
        name: str,
        step: str = constants.NODE + "-0",
        follow: bool | None = False,
    ) -> Iterator[str]:
        """Get logs from a specific step of a TrainJob.

        Args:
            name: The unique name of the TrainJob.
            step: The step of the TrainJob to collect logs from (e.g., 
                "dataset-initializer" or "node-0"). Defaults to "node-0".
            follow: Whether to stream logs in realtime as they are produced. 
                Defaults to False.

        Returns:
            Iterator[str]: An iterator yielding log lines as strings.

        Raises:
            TimeoutError: If the request to retrieve logs times out.
            RuntimeError: If the backend fails to retrieve logs.

        Example:
            from kubeflow.trainer import TrainerClient

            client = TrainerClient()
            for log_line in client.get_job_logs(name="my-job", follow=True):
                print(log_line)
        """
        return self.backend.get_job_logs(name=name, follow=follow, step=step)

    def get_job_events(self, name: str) -> list[types.Event]:
        """Get Kubernetes events associated with a TrainJob.

        Events provide additional clarity about the state of the TrainJob, 
        such as pod state changes, errors, and other significant occurrences.

        Args:
            name: The unique name of the TrainJob.

        Returns:
            list[kubeflow.trainer.types.Event]: A list of Event objects 
                associated with the TrainJob.

        Raises:
            TimeoutError: If the request to retrieve events times out.
            RuntimeError: If the backend fails to retrieve events.

        Example:
            from kubeflow.trainer import TrainerClient

            client = TrainerClient()
            events = client.get_job_events(name="my-job")
            for event in events:
                print(f"Event: {event.message} ({event.reason})")
        """
        return self.backend.get_job_events(name=name)

    def wait_for_job_status(
        self,
        name: str,
        status: set[str] = {constants.TRAINJOB_COMPLETE},
        timeout: int = 600,
        polling_interval: int = 2,
        callbacks: list[Callable[[types.TrainJob], None]] | None = None,
    ) -> types.TrainJob:
        """Wait for a TrainJob to reach a desired status.

        Args:
            name: The unique name of the TrainJob.
            status: A set of expected statuses to wait for (e.g., {"Complete"}). 
                Must be a subset of "Created", "Running", "Complete", and "Failed".
            timeout: Maximum number of seconds to wait. Defaults to 600.
            polling_interval: Seconds to wait between status checks. Defaults to 2.
            callbacks: Optional list of callback functions to invoke after each 
                poll. Each callback accepts the `TrainJob` object as an argument.

        Returns:
            kubeflow.trainer.types.TrainJob: The TrainJob object after reaching 
                the desired status.

        Raises:
            ValueError: If the input values (e.g., status) are invalid.
            RuntimeError: If the job reaches an unexpected "Failed" status or 
                the client fails to get the status.
            TimeoutError: If the job does not reach the desired status within 
                the timeout period.

        Example:
            from kubeflow.trainer import TrainerClient
            from kubeflow.trainer.constants import constants

            client = TrainerClient()
            job = client.wait_for_job_status(
                name="my-job", 
                status={constants.TRAINJOB_COMPLETE}
            )
            print(f"Job finished with status: {job.status}")
        """
        return self.backend.wait_for_job_status(
            name=name,
            status=status,
            timeout=timeout,
            polling_interval=polling_interval,
            callbacks=callbacks,
        )

    def delete_job(self, name: str):
        """Delete a TrainJob.

        Args:
            name: The unique name of the TrainJob to delete.

        Raises:
            TimeoutError: If the request to delete the job times out.
            RuntimeError: If the backend fails to delete the TrainJob.

        Example:
            from kubeflow.trainer import TrainerClient

            client = TrainerClient()
            client.delete_job(name="my-job")
        """
        return self.backend.delete_job(name=name)
