import smtplib
import subprocess as sp
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
# from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
from .error_message import error_traceback

import platform
from datetime import datetime

def sendmail(message, fromaddr='gilyou.public@gmail.com',
             toaddr='gilyou.private@gmail.com', passwd='gilyou1314521',
             subject=None, mtype='plain', screenshot=False, **kwargs):
    """
        send mail
        Attributes
        ----------
        message : string
                  message / mail to send
        fromaddr : string
                   mail address from which to send mail
        toaddr : string
                 mail address to send mail to
        passwd : string
                 password to login to `fromaddr` mail server
        subject : string
                  mail subject
        mtype : string ['plain', 'html', 'markdown']
                message text type
        screenshot : bool
                     attach screenshot or not
        others :
                  NOTE that,  python3-markdown provide basic notations
                  if more notations need, specify extensions
                  For example, to use table notation:
                      message = markdown.markdown(message, extesions=['markdown.extensions.tables'])
                  Supported extensions:
                    |        Extension     |	         “Name”                 |
                    |:--------------------:|------------------------------------|
                    |Extra  	           |    markdown.extensions.extra       |
                    |    Abbreviations     |	markdown.extensions.abbr        |
                    |    Attribute Lists   |	markdown.extensions.attr_list   |
                    |    Definition Lists  |	markdown.extensions.def_list    |
                    |    Fenced Code Blocks|	markdown.extensions.fenced_code |
                    |    Footnotes         |	markdown.extensions.footnotes   |
                    |    Tables            |	markdown.extensions.tables      |
                    |    Smart Strong      |	markdown.extensions.smart_strong|
                    |Admonition            |	markdown.extensions.admonition  |
                    |CodeHilite            |	markdown.extensions.codehilite  |
                    |HeaderId              |	markdown.extensions.headerid    |
                    |Meta-Data             |	markdown.extensions.meta        |
                    |New Line to Break     |	markdown.extensions.nl2br       |
                    |Sane Lists            |	markdown.extensions.sane_lists  |
                    |SmartyPants           |	markdown.extensions.smarty      |
                    |Table of Contents     |	markdown.extensions.toc         |
                    |WikiLinks             |	markdown.extensions.wikilinks   |
                  For more extensions, please see ![python3-markdown extensions]
                  (https://pythonhosted.org/Markdown/extensions/index.html)
    """
    if mtype not in ['plain', 'html', 'markdown']:
        raise ValueError('`mtype` must be one of `plain`,'
                         ' `html` and `markdown`. given {}'
                         .format(mtype))
    if subject is None:
        subject = '{} @ {}'.format(platform.node(),
                          datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    else:
        subject = '{} [{}@{}]'.format(subject, platform.node(),
                          datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    mimepart = MIMEMultipart()
    mimepart['From'] = fromaddr
    mimepart['To'] = toaddr
    mimepart['Subject'] = subject

    if screenshot:
        try:
            filename='/tmp/gnome-screenshot-tmp.png'
            sp.run(['gnome-screenshot', '-f', filename])
            with open(filename, 'rb') as fp:
                attachment = MIMEBase('application', "octet-stream")
                attachment.set_payload(fp.read())
            encoders.encode_base64(attachment)
            attachment.add_header('Content-Disposition', 'attachment',
                                  filename='gnome-screenshot-tmp.png')
            mimepart.attach(attachment)
        except Exception:
            error = "\n{}=> **Error when trying to send screenshot: {}**"\
                    .format('='*5, error_traceback())
            if mtype in ['html', 'markdown']:
                error = error.replace('\n', '<br/>')
            message += error

    if mtype == 'markdown':
        mtype = 'html'
        try:
            import markdown
            message = markdown.markdown(message, **kwargs)
        except Exception:
            error = "\n{}=> Error when trying to makrdown message: {}"\
                    .format('#'*5, error_traceback())
            if mtype in ['html', 'markdown']:
                error = error.replace('\n', '<br/>')
            message += error

    mimepart.attach(MIMEText(message, mtype))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, passwd)
    server.sendmail(fromaddr, toaddr, mimepart.as_string())
    server.quit()

if __name__ == '__main__':
    sendmail('**Hello, world**')
