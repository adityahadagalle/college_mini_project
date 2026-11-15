from app.core.security import create_access_token, decode_access_token, get_password_hash, verify_password


def test_password_hashing():
    password = "secretpassword123"
    hashed = get_password_hash(password)
    assert hashed != password
    assert verify_password(password, hashed) is True
    assert verify_password("wrongpassword", hashed) is False


def test_jwt_token_flow():
    user_id = "test-user-12345"
    token = create_access_token(user_id)
    decoded_user_id = decode_access_token(token)
    assert decoded_user_id == user_id
