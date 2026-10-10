from django.contrib.auth.models import User
from django.test import Client, TestCase


class AccountTests(TestCase):
    def signup_data(self, **changes):
        data = {
            "first_name": "Research",
            "last_name": "Student",
            "username": "student",
            "password1": "Oak-river-research-489!",
            "password2": "Oak-river-research-489!",
        }
        data.update(changes)
        return data

    def test_research_and_download_require_login(self):
        for path in ["/", "/data/prices.csv"]:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 302)
            self.assertTrue(response.url.startswith("/login/?next="))
        self.assertEqual(self.client.get("/health/").status_code, 200)

    def test_signup_saves_names_hashes_password_and_logs_in(self):
        response = self.client.post("/signup/", self.signup_data())
        self.assertRedirects(response, "/", fetch_redirect_response=False)
        user = User.objects.get(username="student")
        self.assertEqual(user.first_name, "Research")
        self.assertEqual(user.last_name, "Student")
        self.assertTrue(user.check_password(self.signup_data()["password1"]))
        self.assertNotEqual(user.password, self.signup_data()["password1"])
        self.assertFalse(user.is_staff)
        self.assertFalse(user.is_superuser)
        self.assertEqual(int(self.client.session["_auth_user_id"]), user.pk)

    def test_duplicate_username_ignores_case(self):
        User.objects.create_user(username="student", password="Existing-student-493!")
        response = self.client.post("/signup/", self.signup_data(username="STUDENT"))
        self.assertEqual(response.status_code, 200)
        self.assertIn("username", response.context["form"].errors)
        self.assertEqual(User.objects.count(), 1)

    def test_signup_rejects_missing_names_or_mismatched_password(self):
        for changes, field in [
            ({"first_name": " "}, "first_name"),
            ({"last_name": ""}, "last_name"),
            ({"password2": "Different-4915!"}, "password2"),
        ]:
            with self.subTest(field=field, changes=changes):
                response = self.client.post("/signup/", self.signup_data(**changes))
                self.assertIn(field, response.context["form"].errors)
        self.assertEqual(User.objects.count(), 0)

    def test_signup_accepts_a_common_short_password(self):
        response = self.client.post("/signup/", self.signup_data(password1="123", password2="123"))
        self.assertRedirects(response, "/", fetch_redirect_response=False)
        self.assertTrue(User.objects.get(username="student").check_password("123"))

    def test_login_error_and_post_only_logout(self):
        User.objects.create_user(username="student", password=self.signup_data()["password1"])
        wrong = self.client.post("/login/", {"username": "student", "password": "incorrect"})
        self.assertContains(wrong, "The username or password is incorrect")
        self.assertNotIn("_auth_user_id", self.client.session)
        right = self.client.post("/login/", {"username": "student", "password": self.signup_data()["password1"]})
        self.assertRedirects(right, "/", fetch_redirect_response=False)
        self.assertEqual(self.client.get("/logout/").status_code, 405)
        self.assertIn("_auth_user_id", self.client.session)
        self.assertRedirects(self.client.post("/logout/"), "/login/", fetch_redirect_response=False)
        self.assertNotIn("_auth_user_id", self.client.session)

    def test_login_rejects_external_redirect(self):
        User.objects.create_user(username="student", password=self.signup_data()["password1"])
        response = self.client.post("/login/?next=https://untrusted.example/", {"username": "student", "password": self.signup_data()["password1"]})
        self.assertRedirects(response, "/", fetch_redirect_response=False)

    def test_signup_login_logout_require_csrf(self):
        client = Client(enforce_csrf_checks=True)
        self.assertEqual(client.post("/signup/", self.signup_data()).status_code, 403)
        self.assertEqual(client.post("/login/", {"username": "student", "password": "wrong"}).status_code, 403)
        user = User.objects.create_user(username="student", password=self.signup_data()["password1"])
        client.force_login(user)
        self.assertEqual(client.post("/logout/").status_code, 403)

    def test_authenticated_users_skip_account_forms(self):
        user = User.objects.create_user(username="student", password=self.signup_data()["password1"])
        self.client.force_login(user)
        for path in ["/login/", "/signup/"]:
            self.assertRedirects(self.client.get(path), "/", fetch_redirect_response=False)

    def test_administrator_can_open_admin_page(self):
        admin_user = User.objects.create_superuser(
            username="administrator", email="admin@example.com", password="admin",
        )
        self.client.force_login(admin_user)
        self.assertContains(self.client.get("/admin/"), "Site administration")
