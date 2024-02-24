# Default parameters are set to run a debug experiment.
set -e

DOMAIN=wmt19.en-de
MODEL=wmt19-en-de
NLINES=3
NSAMPLES=32
EPS=0.01
TOPK=0
TOPP=1.0
SIM=bertscore
EVAL=sacrebleu
ALGORITHM=None
DEBUG=0
RECOMPUTE=""


while getopts d:m:p:l:s:e:k:n:i:v:a:bru:t:z:w:o:h: option
do
  case $option in
    d)
        DOMAIN=${OPTARG};;
    m)
        MODEL=${OPTARG};;
    p)
        PROMPT=${OPTARG};;
    l)
        NLINES=${OPTARG};;
    s)
        NSAMPLES=${OPTARG};;
    e)
        EPS=${OPTARG};;
    k)
        TOPK=${OPTARG};;
    n)
        # Nucleus sampling
        TOPP=${OPTARG};;
    i)
        # TODO: Long options
        SIM=${OPTARG};;
    v)
        EVAL=${OPTARG};;
    a)
        # TODO: Enable arguments for algorithm e.g. k, div_pen
        ALGORITHM=${OPTARG};;
    b)
        DEBUG=1;;
    r)
        RECOMPUTE="--recompute";;
    \?)
      echo "This is unexpected option." 1>&2
      exit 1
  esac
done

# TODO: Refactor this with option
if [ "$DOMAIN" == "xsum" ]; then
    DATADIR=xsum
elif [ "$DOMAIN" == "cnndm" ]; then
    DATADIR=cnn_cln
elif [ "$DOMAIN" == "samsum" ]; then
    DATADIR=None
elif [[ $DOMAIN == "wmt19"* ]]; then
    DATADIR=wmt19-text
elif [ "$DOMAIN" == "iwslt17.fr-en" ]; then
    DATADIR=iwslt17-text
elif [ "$DOMAIN" == "nocaps" ]; then
    DATADIR=None
elif [ "$DOMAIN" == "mscoco" ]; then
    DATADIR=None
elif [ "$DOMAIN" == "mscoco-ft" ]; then
    DATADIR=None
elif [ "$DOMAIN" == "e2e_nlg" ]; then
    DATADIR=None
else
    echo "No cache available for $DOMAIN. Loading from huggingface datasets."
    DATADIR=None
    # exit 1
fi


python3 mbr/mbr_engine.py $DOMAIN \
    --model $MODEL \
    --sample_dir ./samples/$DOMAIN/$MODEL \
    --n_lines $NLINES --n_samples $NSAMPLES \
    --eps $EPS --topk $TOPK --topp $TOPP \
    --sim $SIM \
    --eval $EVAL \
    --algorithm $ALGORITHM \
    $RECOMPUTE \

if [ "$DEBUG" == "1" ]; then
    echo "done!"
fi

