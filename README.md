CS109 Project
=============

## Contributors

Kenny Yu, Ali Nahm, R.J. Aquino, Joseph Ong

## Links

* [Project Proposal](https://docs.google.com/document/d/1UfhnZtgWfwou-JXBBxeVZCXIqhIrCSaDADXmRzBvOsE/edit)

## Scraper Instructions

There are three steps to arriving at the final dataset. First, however, let's start with dependencies. Execute the following (if you're on Ubuntu, else substitute the first line with the appropriate line to install pip).

    sudo apt-get install python-pip
    sudo pip install ntlk
    sudo pip install mrjob
    sudo pip install praw

Additionally, the csplit program needed must be the GNU version, which has the * extension. If you're using the BSD version bundled with MacOS, you're out of luck, though I can try to figure out a workaround soon.

***

1. First, scraping the data from Reddit. To do this, you'll need to use the `scraper.py` file located in the `scraper/scraper` directory. Basic usage is as follows:

        python scraper.py < inputfile > outputfile

    Where input file is a list of subreddits, one per line and surrounding by double quotes. Like this:

        "pics"
        "politics"
        "aww"
        "todayilearned"
        "movies"

    This will start a MapReduce job. It won't work very well unless you have some instances you can spin up on a cluster, or on Amazon EC2. 
    
    You can create a ~/.mrjob.conf file in your home directory that will do all the necessary work to get the script running on EC2 (for example, installing dependencies on the cluster, etc.). I've provided an example in the `scraper/scraper` directory, but **you'll need to fill in your own Amazon credentials**, and then move it into your home directory as via

        mv example.mrjob.conf ~/.mrjob.conf

    You'll notice that conf file has a configuration called "emr". You can then run the script with that configuration as follows:

        python scraper.py -r emr < inputfile > outputfile

    Further, you'll want to optimize this script based on the number of inputs you have. In particular, you'll want the number of mappers to be equivalent to the number of subreddits scraped, to minimize the scraping runtime. For example, I had 24 subreddits to scrape, so I set the number of mapper tasks to 24 as follows:

        python scraper.py -r emr --jobconf mapred.map.tasks=24 < subreddits_important > sr_impt_data

    You also want to make sure that the number of instances you spin up is appropriate for the number of subreddits you are scraping. In this case, I spun up 12 m1.large instances, as per `.mrjob.conf`, so 2 mappers ran per instance. 
    
    Don't run 12 instances for 1 subreddit, or you'll just be wasting a lot of money. On the flipsie, don't use 1 instance for 24 subreddits, or your job will run for an excrutiatingly long time. 
    
    I've found that 2 subreddits per instance works well, and will make your scraping job last about 6 hours.

    I should really handle a lot of this under the hood for you, but at the moment, I don't think mr.job has hooks to let me dynamically set settings based on input size. Sorry!

2. Next, you'll need to clean the data you get back from the MapReduce job. In the `scraper/cleaner` directory, you'll find a clean.py script. It's also a MR script, but you can probably just run this one on your computer instead of a cluster -- it shouldn't take too long unless step #1 produced a huge file (I mean like, > 5 GB).

        python cleaner.py < unclean_input > cleaned_output

    Where unclean\_input is the output of step #1, and clean\_name is the name of the clean file you want.


3. Finally, we'll need to segment the huge chunk of data we scraped into individual subreddit files. In order to do this, look in the `scraper/segmenter` directory. You can then run the command:

        ./segment inputfile outputdir
        
    Where inputfile is the file produced from step #2, and outputdir is the name of the directory to which you'll want to put all the new information. You'll have to make the directory beforehand.
