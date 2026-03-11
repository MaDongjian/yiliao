from flask import Blueprint,request,send_file
from flask_apispec import use_kwargs
from utils.rest_response import success_response
import pandas as pd
from datetime import datetime,timedelta
import io
from urllib.parse import quote




test_blueprint = Blueprint('test', __name__)


@test_blueprint.route("/test", methods=["GET"])
@use_kwargs({}, location='querystring')
def test_func():
    return success_response(data="cg")

@test_blueprint.route("/test11", methods=["GET"])
@use_kwargs({}, location='querystring')
def test_func11():
    return success_response(data="cg")







