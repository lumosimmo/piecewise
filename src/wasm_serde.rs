use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub fn deserialize_u64<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    match value {
        serde_json::Value::Number(n) => n
            .as_u64()
            .or_else(|| n.as_f64().map(|f| f as u64))
            .ok_or_else(|| serde::de::Error::custom("Invalid u64 value")),
        serde_json::Value::String(s) => s
            .parse::<u64>()
            .map_err(|_| serde::de::Error::custom("Invalid u64 string")),
        _ => Err(serde::de::Error::custom("Expected number or string")),
    }
}

pub fn deserialize_option_u128<'de, D>(deserializer: D) -> Result<Option<u128>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    match value {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Number(n) => {
            let num = n
                .as_u64()
                .map(|u| u as u128)
                .or_else(|| n.as_f64().map(|f| f as u128))
                .ok_or_else(|| serde::de::Error::custom("Invalid u128 value"))?;
            Ok(Some(num))
        }
        serde_json::Value::String(s) => {
            let num = s
                .parse::<u128>()
                .map_err(|_| serde::de::Error::custom("Invalid u128 string"))?;
            Ok(Some(num))
        }
        _ => Err(serde::de::Error::custom("Expected number, string, or null")),
    }
}

pub fn serialize_usize<S>(value: &usize, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    (*value as f64).serialize(serializer)
}
