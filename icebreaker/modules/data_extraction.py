"""Extract LinkedIn profile data from Proxycurl or the sample dataset."""

from typing import Any, Dict, Optional

import requests

import config


def _remove_empty_values(value: Any) -> Any:
    """Remove empty values recursively while preserving meaningful zeroes."""

    if isinstance(value, dict):
        return {
            key: cleaned
            for key, item in value.items()
            if (cleaned := _remove_empty_values(item)) not in (None, "", [], {})
        }
    if isinstance(value, list):
        return [cleaned for item in value if (cleaned := _remove_empty_values(item)) not in (None, "", [], {})]
    return value


def _get_json(url: str, **kwargs: Any) -> Dict[str, Any]:
    try:
        response = requests.get(url, timeout=30, **kwargs)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError("Could not download LinkedIn profile data.") from exc
    except ValueError as exc:
        raise RuntimeError("The profile data response was not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("The profile data response had an unexpected format.")
    return payload


def extract_linkedin_profile(
    linkedin_profile_url: str,
    api_key: Optional[str] = None,
    mock: bool = False,
) -> Dict[str, Any]:
    """Load a profile from the sample URL or Proxycurl."""

    if mock:
        profile = _get_json(config.MOCK_DATA_URL)
    else:
        if not linkedin_profile_url:
            raise ValueError("A LinkedIn profile URL is required.")
        if not api_key:
            raise ValueError("A Proxycurl API key is required for live profiles.")
        profile = _get_json(
            "https://nubela.co/proxycurl/api/v2/linkedin",
            headers={"Authorization": f"Bearer {api_key}"},
            params={"url": linkedin_profile_url},
        )

    cleaned_profile = _remove_empty_values(profile)
    if not cleaned_profile:
        raise ValueError("The LinkedIn profile contains no usable data.")
    return cleaned_profile
