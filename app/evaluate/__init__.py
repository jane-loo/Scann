from flask import Blueprint

evaluate_bp = Blueprint('evaluate', __name__)

from . import routes
