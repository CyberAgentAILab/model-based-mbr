DOMAIN=wmt19.en-de
MODEL=None # Use "None" for using the sequence-to-sequence models.
PROMPT=None
NLINES=3
NSAMPLES=32

EPS=0.01
TOPK=0
TOPP=1.0 # nucleus

BSZ=16

while getopts d:m:p:l:s:e:k:n:t:z: option
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
        TOPP=${OPTARG};;
    z)
        BSZ=${OPTARG};;
    \?)
      echo "This is unexpected option." 1>&2
      exit 1
  esac
done

if [ "$DOMAIN" == "xsum" ]; then
    DATADIR=xsum
elif [ "$DOMAIN" == "cnndm" ]; then
    DATADIR=cnn_cln
elif [ "$DOMAIN" == "samsum" ]; then
    DATADIR=None
elif [[ $DOMAIN == "wmt19"* ]]; then
    DATADIR=wmt19-text
elif [ "$DOMAIN" == "nocaps" ]; then
    DATADIR=None
elif [ "$DOMAIN" == "mscoco-ft" ]; then
    DATADIR=None
elif [ "$DOMAIN" == "e2e_nlg" ]; then
    DATADIR=None
else
    echo "No cache available for dataset: $DOMAIN. Downloading from huggingface datasets."
    DATADIR=None
fi


echo "sampling..."
python3 mbr/sample.py $DOMAIN \
    --model $MODEL --prompt $PROMPT \
    --n_lines $NLINES --n_samples $NSAMPLES \
    --eps $EPS --topk $TOPK --topp $TOPP \
    --bsz $BSZ