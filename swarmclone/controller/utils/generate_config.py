import secrets
import tomlkit

def generate_config():
    secret_key = secrets.token_urlsafe(32)
    config = {
        'app': {
            'secret_key': secret_key,
            'algorithm': 'HS256',
            'access_token_expire_minutes': 30,
            'refresh_token_expire_days': 7,
            'database_url': 'sqlite:///./data.db'
        }
    }
    with open('./dist/server_config.toml', 'w') as f:
        tomlkit.dump(config, f)

if __name__ == "__main__":
    generate_config()