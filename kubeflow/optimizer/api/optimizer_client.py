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

from collections.abc import Callable, Iterator
import logging
from typing import Any

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.optimizer.backends.kubernetes.backend import KubernetesBackend
from kubeflow.optimizer.constants import constants
from kubeflow.optimizer.types.algorithm_types import BaseAlgorithm
from kubeflow.optimizer.types.optimization_types import (
    Objective,
    OptimizationJob,
    Result,
    TrialConfig,
)
from kubeflow.trainer.types.types import Event, TrainJobTemplate

logger = logging.getLogger(__name__)


class OptimizerClient:
    def __init__(
        self,
        backend_config: KubernetesBackendConfig | None = None,
    ):
        """Initialize a Kubeflow Optimizer client.

        Args:
            backend_config: Backend configuration. Either a `KubernetesBackendConfig` 
                or `None` to use the default config class. Defaults to 
                `KubernetesBackendConfig`.

        Raises:
            ValueError: If the provided `backend_config` is invalid.

        Example:
            from kubeflow.optimizer import OptimizerClient

            client = OptimizerClient()
        """
        # Set the default backend config.
        if not backend_config:
            backend_config = KubernetesBackendConfig()

        if isinstance(backend_config, KubernetesBackendConfig):
            self.backend = KubernetesBackend(backend_config)
        else:
            raise ValueError(f"Invalid backend config '{backend_config}'")

    def optimize(
        self,
        trial_template: TrainJobTemplate,
        *,
        trial_config: TrialConfig | None = None,
        search_space: dict[str, Any],
        objectives: list[Objective] | None = None,
        algorithm: BaseAlgorithm | None = None,
    ) -> str:
        """Create and start an OptimizationJob for hyperparameter tuning.

        Args:
            trial_template: The TrainJob template defining the training code.

        Keyword Args:
            trial_config: Optional configuration for running Trials (e.g., resources,
                parallelism).
            search_space: A dictionary mapping parameter names to search 
                specifications. Use `Search.uniform()`, `Search.loguniform()`, 
                `Search.choice()`, etc.
            objectives: A list of objectives to optimize (e.g., maximizing accuracy).
            algorithm: The optimization algorithm to use (e.g., `RandomSearch`, 
                `BayesianOptimization`). Defaults to `RandomSearch`.

        Returns:
            str: The unique name of the generated OptimizationJob (Experiment).

        Raises:
            ValueError: If the input arguments are invalid.
            TimeoutError: If the request to create the job times out.
            RuntimeError: If the backend fails to create the job.

        Example:
            from kubeflow.optimizer import OptimizerClient, Search, Objective

            client = OptimizerClient()

            # Define search space
            search_space = {
                "learning_rate": Search.loguniform(min=0.01, max=0.1),
                "batch_size": Search.choice(values=[16, 32, 64]),
                "optimizer": Search.choice(values=["sgd", "adam"])
            }

            # Define objective
            objectives = [Objective(metric="accuracy", direction="maximize")]

            # Assumes `my_template` is a TrainJobTemplate defined elsewhere
            job_name = client.optimize(
                trial_template=my_template,
                search_space=search_space,
                objectives=objectives
            )
            print(f"Started optimization job: {job_name}")
        """
        return self.backend.optimize(
            trial_template=trial_template,
            trial_config=trial_config,
            objectives=objectives,
            search_space=search_space,
            algorithm=algorithm,
        )

    def list_jobs(self) -> list[OptimizationJob]:
        """List the created OptimizationJobs.

        Returns:
            list[kubeflow.optimizer.types.OptimizationJob]: A list of created 
                OptimizationJobs. If no jobs exist, an empty list is returned.

        Raises:
            TimeoutError: If the request to list jobs times out.
            RuntimeError: If the backend fails to list jobs.

        Example:
            from kubeflow.optimizer import OptimizerClient

            client = OptimizerClient()
            jobs = client.list_jobs()
            for job in jobs:
                print(f"Job Name: {job.name}, Status: {job.status}")
        """

        return self.backend.list_jobs()

    def get_job(self, name: str) -> OptimizationJob:
        """Get information about a specific OptimizationJob.

        Args:
            name: The unique name of the OptimizationJob.

        Returns:
            kubeflow.optimizer.types.OptimizationJob: The OptimizationJob object.

        Raises:
            TimeoutError: If the request to retrieve the job times out.
            RuntimeError: If the backend fails to retrieve the job.

        Example:
            from kubeflow.optimizer import OptimizerClient

            client = OptimizerClient()
            job = client.get_job(name="my-opt-job-abc123")
            print(f"Job Status: {job.status}")
        """

        return self.backend.get_job(name=name)

    def get_job_logs(
        self,
        name: str,
        trial_name: str | None = None,
        follow: bool = False,
    ) -> Iterator[str]:
        """Get logs from a specific trial of an OptimizationJob.

        Args:
            name: The unique name of the OptimizationJob.
            trial_name: Optional name of a specific Trial. If not provided, logs 
                from the current best trial are returned. If no best trial is 
                available, logs from the first trial are returned.
            follow: Whether to stream logs in realtime as they are produced. 
                Defaults to False.

        Returns:
            Iterator[str]: An iterator yielding log lines as strings.

        Raises:
            TimeoutError: If the request to retrieve logs times out.
            RuntimeError: If the backend fails to retrieve logs.

        Example:
            from kubeflow.optimizer import OptimizerClient

            client = OptimizerClient()

            # Get logs from the current best trial
            for log_line in client.get_job_logs(name="my-opt-job"):
                print(log_line)

            # Stream logs from a specific trial
            for log_line in client.get_job_logs(
                name="my-opt-job", 
                trial_name="my-opt-job-trial-1", 
                follow=True
            ):
                print(log_line)
        """
        return self.backend.get_job_logs(name=name, trial_name=trial_name, follow=follow)

    def get_best_results(self, name: str) -> Result | None:
        """Get the best hyperparameters and metrics from an OptimizationJob.

        This method retrieves the optimal hyperparameters and their corresponding 
        metrics from the best trial found during the optimization process.

        Args:
            name: The unique name of the OptimizationJob.

        Returns:
            Optional[kubeflow.optimizer.types.Result]: An object containing the 
                best hyperparameters and metrics, or `None` if no best trial is 
                available yet.

        Raises:
            TimeoutError: If the request to retrieve results times out.
            RuntimeError: If the backend fails to retrieve results.

        Example:
            from kubeflow.optimizer import OptimizerClient

            client = OptimizerClient()
            best_result = client.get_best_results(name="my-opt-job")
            if best_result:
                print(f"Best parameters: {best_result.parameters}")
                print(f"Best metrics: {best_result.metrics}")
        """
        return self.backend.get_best_results(name=name)

    def wait_for_job_status(
        self,
        name: str,
        status: set[str] = {constants.OPTIMIZATION_JOB_COMPLETE},
        timeout: int = 3600,
        polling_interval: int = 2,
        callbacks: list[Callable[[OptimizationJob], None]] | None = None,
    ) -> OptimizationJob:
        """Wait for an OptimizationJob to reach a desired status.

        Args:
            name: The unique name of the OptimizationJob.
            status: A set of expected statuses to wait for (e.g., {"Complete"}). 
                Must be a subset of "Created", "Running", "Complete", and "Failed".
            timeout: Maximum number of seconds to wait. Defaults to 3600.
            polling_interval: Seconds to wait between status checks. Defaults to 2.
            callbacks: Optional list of callback functions to invoke after each poll. 
                Each callback accepts the `OptimizationJob` object as an argument.

        Returns:
            kubeflow.optimizer.types.OptimizationJob: The OptimizationJob object 
                after reaching the desired status.

        Raises:
            ValueError: If the input values (e.g., status) are invalid.
            RuntimeError: If the job reaches an unexpected "Failed" status or 
                the client fails to get the status.
            TimeoutError: If the job does not reach the desired status within 
                the timeout period.

        Example:
            from kubeflow.optimizer import OptimizerClient
            from kubeflow.optimizer.constants import constants

            client = OptimizerClient()
            job = client.wait_for_job_status(
                name="my-opt-job", 
                status={constants.OPTIMIZATION_JOB_COMPLETE}
            )
            print(f"Optimization finished with status: {job.status}")
        """
        return self.backend.wait_for_job_status(
            name=name,
            status=status,
            timeout=timeout,
            polling_interval=polling_interval,
            callbacks=callbacks,
        )

    def delete_job(self, name: str):
        """Delete an OptimizationJob.

        Args:
            name: The unique name of the OptimizationJob to delete.

        Raises:
            TimeoutError: If the request to delete the job times out.
            RuntimeError: If the backend fails to delete the job.

        Example:
            from kubeflow.optimizer import OptimizerClient

            client = OptimizerClient()
            client.delete_job(name="my-opt-job")
        """
        return self.backend.delete_job(name=name)

    def get_job_events(self, name: str) -> list[Event]:
        """Get Kubernetes events associated with an OptimizationJob.

        Events provide additional clarity about the state of the OptimizationJob, 
        such as trial state changes, errors, and other significant occurrences.

        Args:
            name: The unique name of the OptimizationJob.

        Returns:
            list[kubeflow.trainer.types.Event]: A list of Event objects 
                associated with the OptimizationJob.

        Raises:
            TimeoutError: If the request to retrieve events times out.
            RuntimeError: If the backend fails to retrieve events.

        Example:
            from kubeflow.optimizer import OptimizerClient

            client = OptimizerClient()
            events = client.get_job_events(name="my-opt-job")
            for event in events:
                print(f"Event: {event.message} ({event.reason})")
        """
        return self.backend.get_job_events(name=name)
