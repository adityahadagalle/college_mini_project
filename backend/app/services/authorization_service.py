from typing import Dict, Set

from app.models.database import Role, User


class AuthorizationService:
    """Centralized RBAC permission service.

    Permissions are stored in a dictionary mapping role names to a set of permission strings.
    """

    _permissions: Dict[Role, Set[str]] = {
        Role.ADMIN: {
            "users:create",
            "users:update",
            "users:delete",
            "users:view",
            "predictions:create",
            "predictions:view",
            "batch:create",
            "batch:view",
            "analytics:view",
            "history:view",
            "settings:update",
            "audit:view",
            "sessions:manage",
        },
        Role.ANALYST: {
            "predictions:create",
            "predictions:view",
            "batch:create",
            "batch:view",
            "analytics:view",
            "history:view",
            "explainability:view",
        },
        Role.USER: {
            "predictions:create",
            "predictions:view",
            "history:view",
            "explainability:view",
        },
    }

    def has_permission(self, user: User, permission: str) -> bool:
        """Return ``True`` if the user's role includes the given permission.

        Args:
            user: The ``User`` model instance.
            permission: Permission string, e.g. ``"users:create"``.
        """
        role_permissions = self._permissions.get(user.role, set())
        return permission in role_permissions
