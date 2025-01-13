from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from .user_service import UserService, get_db
from .auth_service import AuthService
import uvicorn
from datetime import timedelta

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

print("[__main__.py] FastAPI application initialized.")
@app.post("/token", response_model=dict)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user_service = UserService(db)
    auth_service = AuthService()  # Create an instance of AuthService
    user = user_service.get_user(form_data.username)
    if not user or not auth_service.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=auth_service.settings.access_token_expire_minutes)
    access_token = auth_service.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    refresh_token = auth_service.create_refresh_token(
        data={"sub": user.username}, expires_delta=timedelta(days=auth_service.settings.refresh_token_expire_days)
    )
    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}

@app.post("/refresh", response_model=dict)
async def refresh_token(refresh_token: str = Depends(oauth2_scheme)):
    print("[__main__.py] /refresh endpoint called.")
    print(f"[__main__.py] Decoding refresh token: {refresh_token}")
    payload = AuthService().decode_token(refresh_token)
    if payload is None or payload.get("type") != "refresh":
        print("[__main__.py] Invalid refresh token.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    print(f"[__main__.py] Refresh token decoded successfully: {payload}")
    access_token = AuthService().create_access_token(data={"sub": payload.get("sub")})
    print(f"[__main__.py] New access token created: {access_token}")
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=dict)
async def read_users_me(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    print("[__main__.py] /users/me endpoint called.")
    print(f"[__main__.py] Decoding token: {token}")
    payload = AuthService().decode_token(token)
    if payload is None:
        print("[__main__.py] Invalid token.")
        raise HTTPException(status_code=401, detail="Invalid token")
    username = payload.get("sub")
    print(f"[__main__.py] Fetching user: {username}")
    user_service = UserService(db)
    user = user_service.get_user(username)
    if not user:
        print(f"[__main__.py] User not found: {username}")
        raise HTTPException(status_code=404, detail="User not found")
    print(f"[__main__.py] User found: {user.username}")
    return {"username": user.username, "disabled": user.disabled}

if __name__ == "__main__":
    print("[__main__.py] Starting FastAPI server...")
    uvicorn.run(app, host="127.0.0.1", port=8080)