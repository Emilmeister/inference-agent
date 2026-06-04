"""Inference benchmark REST API service.

Exposes experiment storage and history queries over HTTP so that the agent
process never holds DB credentials directly. The agent talks to this service
via `inference_agent.api_client.ExperimentApiClient`.
"""
