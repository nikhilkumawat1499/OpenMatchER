from openmatcher_matching import MatchingConfig, run_resolution

records = [
    {"name": "Acme Inc", "address": "10 Market Street"},
    {"name": "ACME Incorporated", "address": "10 Market St"},
    {"name": "Globex LLC", "address": "500 Lake Road"},
]

result = run_resolution(records, MatchingConfig(name_column="name", threshold=0.82))
print(result["metrics"])
print(result["matches"][:3])

