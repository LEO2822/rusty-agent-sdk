use reqwest::StatusCode;
use std::time::Duration;

pub fn is_retryable_status(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::TOO_MANY_REQUESTS
            | StatusCode::INTERNAL_SERVER_ERROR
            | StatusCode::BAD_GATEWAY
            | StatusCode::SERVICE_UNAVAILABLE
            | StatusCode::GATEWAY_TIMEOUT
    )
}

pub fn is_retryable_error(error: &reqwest::Error) -> bool {
    error.is_timeout() || error.is_connect() || error.is_request()
}

pub fn retry_delay(base: Duration, attempt: u32) -> Duration {
    let multiplier = 1_u32 << attempt.min(8);
    base.saturating_mul(multiplier)
}
