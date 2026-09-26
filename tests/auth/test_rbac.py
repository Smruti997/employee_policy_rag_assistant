from app.auth.models import AuthenticatedUser


def test_user_receives_company_hr_and_own_department_scope() -> None:
    user = AuthenticatedUser(
        user_id="emp-001",
        email="emp@test.com",
        department="finance",
        level=1,
    )

    assert user.access_scope.departments == ("hr", "finance")
    assert user.access_scope.qdrant_filter() == {
        "must": [
            {"key": "department", "match": {"any": ["hr", "finance"]}},
            {"key": "access_level", "range": {"lte": 1}},
        ]
    }


def test_hr_user_scope_does_not_duplicate_hr() -> None:
    user = AuthenticatedUser(
        user_id="mgr-002",
        email="mgr@test.com",
        department="hr",
        level=2,
    )

    assert user.access_scope.departments == ("hr",)
    assert user.access_scope.maximum_access_level == 2
