FROM gcr.io/google_appengine/python
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# virtualenv
RUN virtualenv /venv -p python3.6
ENV VIRTUAL_ENV /venv
ENV PATH /venv/bin:$PATH


COPY myproject/requirements.txt .

RUN pip3 install -r requirements.txt

RUN python3 -m spacy download en_core_web_sm

# Copy code
COPY myproject/ .

EXPOSE $PORT
CMD gunicorn -b :$PORT main:app