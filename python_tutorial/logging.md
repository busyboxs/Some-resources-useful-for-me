# Logging HowTO

## Logging function

* [`debug()`](https://docs.python.org/2/library/logging.html#logging.debug)
* [`info()`](https://docs.python.org/2/library/logging.html#logging.info)
* [`warning()`](https://docs.python.org/2/library/logging.html#logging.warning)
* [`error()`](https://docs.python.org/2/library/logging.html#logging.error)
* [`critical()`](https://docs.python.org/2/library/logging.html#logging.critical)

## When to use logging

|Task you want to perform | The best tool for task|
|-------------------------|------------------------|
|Display console output for ordinary usage of a command line script or program|[`print()`](https://docs.python.org/2/library/functions.html#print)|
|Report events that occur during normal operation of a program (e.g. for status monitoring or fault investigation)|[`logging.info()`](https://docs.python.org/2/library/logging.html#logging.info) (or [`logging.debug()`](https://docs.python.org/2/library/logging.html#logging.debug) for very detailed output for diagnostic purposes)|
|Issue a warning regarding a particular runtime event| [`warnings.warn()`](https://docs.python.org/2/library/warnings.html#warnings.warn) in library code if the issue is avoidable and the client application should be modified to eliminate the warning; [`logging.warning()`](https://docs.python.org/2/library/logging.html#logging.warning) if there is nothing the client application can do about the situation, but the event should still be noted|
|Report an error regarding a particular runtime event| Raise an exception|
|Report suppression of an error without raising an exception (e.g. error handler in a long-running server process)|[`logging.error()`](https://docs.python.org/2/library/logging.html#logging.error), [`logging.exception()`](https://docs.python.org/2/library/logging.html#logging.exception) or [`logging.critical()`](https://docs.python.org/2/library/logging.html#logging.critical) as appropriate for the specific error and application domain|

## Logging level

|Level|When it's used|
|-----|--------------|
|`DEBUG`|	Detailed information, typically of interest only when diagnosing problems.|
|`INFO`|	Confirmation that things are working as expected.|
|`WARNING`(default)|	An indication that something unexpected happened, or indicative of some problem in the near future (e.g. ‘disk space low’). The software is still working as expected.|
|`ERROR`|	Due to a more serious problem, the software has not been able to perform some function.|
|`CRITICAL`|	A serious error, indicating that the program itself may be unable to continue running.|

The default level is WARNING, which means that only events of this level and above will be tracked, unless the logging package is configured to do otherwise.

> DEBUG > INFO > WARNING > ERROR > CRITICAL
> This means that, if the level is WARNING, then WARNING, ERROR, CRITICAL information will be tracked; If the level is INFO, then INFO, WARNING, ERROR and CRITICAL infomation will be tracked.

Events that are tracked can be handled in different ways. The simplest way of handling tracked events is to **print them to the console**. Another common way is to **write them to a disk file**.

## A simple example

```python
import logging
logging.warning('Watch out!')  # will print a message to the console
logging.info('I told you so')  # will not print anything
```

output:

```
WARNING:root:Watch out!
```

## Logging to a file

```python
import logging
logging.basicConfig(filename='example.log',level=logging.DEBUG)
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')
```

output:

```
DEBUG:root:This message should go to the log file
INFO:root:So should this
WARNING:root:And this, too
```

If you run the above script several times, the messages from successive runs are appended to the file example.log. If you want each run to start afresh, not remembering the messages from earlier runs, you can specify the filemode argument, by changing the call in the above example to:

```python
logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG)
```

The output will be the same as before, but the log file is no longer appended to, so the messages from earlier runs are lost.


## Logging from multiple modules

```python
# myapp.py
import logging
import mylib

def main():
    logging.basicConfig(filename='myapp.log', level=logging.INFO)
    logging.info('Started')
    mylib.do_something()
    logging.info('Finished')

if __name__ == '__main__':
    main()
```

```python
# mylib.py
import logging

def do_something():
    logging.info('Doing something')
```

output of file:

```
INFO:root:Started
INFO:root:Doing something
INFO:root:Finished
```

## Logging variable data

```python
import logging
logging.warning('%s before you %s', 'Look', 'leap!')
```

output:

```
WARNING:root:Look before you leap!
```

## Changing the format of displayed messages

```python
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logging.debug('This message should appear on the console')
logging.info('So should this')
logging.warning('And this, too')
```

output:

```
DEBUG:This message should appear on the console
INFO:So should this
WARNING:And this, too
```

## Displaying the date/time in messages

### ISO8601(default) format

```python
import logging
logging.basicConfig(format='%(asctime)s %(message)s')
logging.warning('is when this event was logged.')
```

output:

```
2018-03-20 14:07:19,536 is when this event was logged.
```

### particular format

```python
import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logging.warning('is when this event was logged.')
```

output

```
03/20/2018 02:10:09 PM is when this event was logged.
```


