from django.test import TestCase

from teenplay.models import TeenPlay


class TeenplayTest(TestCase):
    teenplay_id_value = list(TeenPlay.enable_objects.values('id'))
    print(TeenPlay.enable_objects.values('id'))