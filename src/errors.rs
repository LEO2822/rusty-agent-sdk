use pyo3::PyErr;
use pyo3::exceptions::{PyConnectionError, PyRuntimeError, PyValueError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SdkError {
    Connection(String),
    Runtime(String),
    Value(String),
}

impl SdkError {
    pub fn connection(message: impl Into<String>) -> Self {
        Self::Connection(message.into())
    }

    pub fn runtime(message: impl Into<String>) -> Self {
        Self::Runtime(message.into())
    }

    pub fn value(message: impl Into<String>) -> Self {
        Self::Value(message.into())
    }

    pub fn into_pyerr(self) -> PyErr {
        match self {
            Self::Connection(message) => PyConnectionError::new_err(message),
            Self::Runtime(message) => PyRuntimeError::new_err(message),
            Self::Value(message) => PyValueError::new_err(message),
        }
    }
}
