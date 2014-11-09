<?php

/**
 * This takes an existing CSV file & cleans it up so an ARFF can be generated
 * @author  Jason Feriante
 */
set_time_limit(0);
require('debug.php');
// $filename = 'yelp_academic_dataset_review.csv';
// $filename = 'yelp_academic_dataset_business.csv';
// $filename = 'yelp_academic_dataset_checkin.csv'; // Doesn't seem useful.
// $filename = 'yelp_academic_dataset_tip.csv';
// $filename = 'yelp_academic_dataset_user.csv';
$outfile = 'data/cleaned-csv/' . $filename;
$infile = 'data/csv/'  . $filename;

// Changes to users: removed type & name
//  -elite becomes count of years they were elite instead of array of actual years
//  -friends just becomes a count of how many friends this user has (a graph representation would be expensive)
function generate_user_csv($infile, $outfile)
{
  // sanitize: name, city, type, 
  $handle = fopen($infile, "r");  
  $handle_out = fopen($outfile, "w"); 
  
  // First line
  $attributes = fgetcsv($handle, 0, ",",'"');
  // echo Debug::vars($attributes); exit;
  unset($attributes[8]);
  unset($attributes[15]);
  $rs = fputcsv($handle_out, $attributes);

  $count = 0;
  while(($line = fgetcsv($handle, 0, ",",'"')) !== false) 
  {
    unset($line[8]);
    unset($line[15]);

    if(strlen($line[14]) > 4)
    {
      $line[14] = count(explode(",", $line[14]));
    }
    else
    {
      $line[14] = '';
    }

    if(strlen($line[3]) > 4)
    {
      $line[3] = count(explode(",", $line[3]));
    }
    else
    {
      $line[3] = '';
    }

    $date = DateTime::createFromFormat('Y-m-d H:i:s', $line[0].'-01 00:00:00');
    //yyyy-MM-dd HH:mm:ss
    //"2001-04-03 12:12:12"
    $line[0] = $date->format('Y-m-d H:i:s');
    $line[16] = '"'.$line[16].'"';
    // echo Debug::vars($line); exit;
    $count++;
    $rs = fputcsv($handle_out, $line);
  }
  if (!feof($handle)) {
    echo "Error: unexpected fgets() fail\n";
  }
  fclose($handle);
  fclose($handle_out);

  $contents = file_get_contents($outfile);
  $contents = str_replace('"""', '"', $contents);
  file_put_contents($outfile, $contents);
}

echo "<pre>starting task\n</pre>";
generate_user_csv($infile, $outfile);
// generate_business_csv($infile, $outfile);
echo "<pre>All done! Generated: $outfile \n</pre>";

// Changes to checkin: removed type
// Changes to tips: removed text, & type
// Changes to reviews: removed text & type


// Changes to Business!! (columns I threw out)
// unset($attributes[9]); // name
// unset($attributes[22]); // categories: this seems extremely important. need better way to flatten!
// unset($attributes[39]); // state
// unset($attributes[46]); // full address
// unset($attributes[61]); // city
// unset($attributes[67]); // type
// unset($attributes[99]); // neighborhoods