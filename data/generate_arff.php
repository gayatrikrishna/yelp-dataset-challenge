<?php

set_time_limit(0);
require('debug.php');
// $filename = 'yelp_academic_dataset_review.csv';
// $filename = 'yelp_academic_dataset_business.csv';
$filename = 'yelp_academic_dataset_checkin.csv'; // Doesn't seem useful.
// $filename = 'yelp_academic_dataset_tip.csv';
// $filename = 'yelp_academic_dataset_user.csv';
$outfile = 'data/cleaned-csv/' . $filename;
$infile = 'data/csv/'  . $filename;

function sort_array_by_keys(array $unsorted, array $sortKeys)
{
  if(count($unsorted) != count($sortKeys))
  {
    echo Debug::vars($unsorted); exit;
  }
  $sorted = [];
  foreach ($sortKeys as $key => $value)
  {
    $sorted[$key] = str_replace(",", ";", $unsorted[$key]);
  }

  return $sorted;
}

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
  asort($attributes);
  // Remove name & type
  unset($attributes[8]);
  unset($attributes[15]);
  $rs = fputcsv($handle_out, $attributes);

  while(($line = fgetcsv($handle, 0, ",",'"')) !== false) 
  {
    // Remove name & type
    unset($line[8]);
    unset($line[15]);

    $line = sort_array_by_keys($line, $attributes);

    if(strlen($line[14]) > 4)
    {
      $line[14] = count(explode(",", $line[14]));
    }
    else
    {
      $line[14] = 0;
    }

    if(strlen($line[3]) > 4)
    {
      $line[3] = count(explode(",", $line[3]));
    }
    else
    {
      $line[3] = 0;
    }

    $date = DateTime::createFromFormat('Y-m-d H:i:s', $line[0].'-01 00:00:00');
    //yyyy-MM-dd HH:mm:ss
    //"2001-04-03 12:12:12"
    $line[0] = $date->format('Y-m-d H:i:s');
    $line[16] = '"'.$line[16].'"';
    $rs = fputcsv($handle_out, $line);
  }
  // if (!feof($handle)) {
  //  echo "Error: unexpected fgets() fail\n";
  // }
  fclose($handle);
  fclose($handle_out);

  $contents = file_get_contents($outfile);
  $contents = str_replace('"""', '"', $contents);
  file_put_contents($outfile, $contents);
}

function generate_business_csv($infile, $outfile)
{
  // sanitize: name, city, type, 
  $handle = fopen($infile, "r");  
  $handle_out = fopen($outfile, "w"); 
  
  // First line
  $attributes = fgetcsv($handle, 0, ",",'"');
  asort($attributes);
  $rs = fputcsv($handle_out, $attributes);

  while(($line = fgetcsv($handle, 0, ",",'"')) !== false) 
  {
    $line = sort_array_by_keys($line, $attributes);
    $rs = fputcsv($handle_out, $line);
  }

  fclose($handle);
  fclose($handle_out);

  $contents = file_get_contents($outfile);
  $contents = str_replace('"""', '"', $contents);
  file_put_contents($outfile, $contents);
}

// CSV to clean CSV
function generate_review($infile, $outfile)
{
  $handle = fopen($infile, "r");  
  $handle_out = fopen($outfile, "w"); 
  
  // First line
  $attributes = fgetcsv($handle, 0, ",",'"');
  asort($attributes);
  $rs = fputcsv($handle_out, $attributes);

  while(($line = fgetcsv($handle, 0, ",",'"')) !== false) 
  {
    $line = sort_array_by_keys($line, $attributes);
    $rs = fputcsv($handle_out, $line);
  }

  fclose($handle);
  fclose($handle_out);

  $contents = file_get_contents($outfile);
  $contents = str_replace('"""', '"', $contents);
  file_put_contents($outfile, $contents);
}

// CSV to clean CSV
function generate_tip($infile, $outfile)
{
  $handle = fopen($infile, "r");  
  $handle_out = fopen($outfile, "w"); 
  
  // First line
  $attributes = fgetcsv($handle, 0, ",",'"');
  asort($attributes);
  $rs = fputcsv($handle_out, $attributes);

  while(($line = fgetcsv($handle, 0, ",",'"')) !== false) 
  {
    $line = sort_array_by_keys($line, $attributes);
    $rs = fputcsv($handle_out, $line);
  }

  fclose($handle);
  fclose($handle_out);

  $contents = file_get_contents($outfile);
}

// CSV to clean CSV
function generate_checkin($infile, $outfile)
{
  $handle = fopen($infile, "r");  
  $handle_out = fopen($outfile, "w"); 
  
  // First line
  $attributes = fgetcsv($handle, 0, ",",'"');
  asort($attributes);
  unset($attributes[103]);
  $rs = fputcsv($handle_out, $attributes);

  while(($line = fgetcsv($handle, 0, ",",'"')) !== false) 
  {
    unset($line[103]);
    $line = sort_array_by_keys($line, $attributes);
    $rs = fputcsv($handle_out, $line);
  }

  fclose($handle);
  fclose($handle_out);

  $contents = file_get_contents($outfile);
}

// JSON to csv
function generate_review_csv($infile, $outfile)
{
  $infile = str_replace(".csv", ".json", $infile);
  $handle = fopen($infile, "r");  
  $handle_out = fopen($outfile, "w"); 
  
  // First line
  $json = fgets($handle);
  $data = json_decode($json);
  
  $attributes = ['votes.funny', 'votes.useful', 'votes.cool', 'user_id', 'review_id', 'stars', 'date', 'business_id'];
  $rs = fputcsv($handle_out, $attributes);

  $date = DateTime::createFromFormat('Y-m-d H:i:s', $data->date.' 00:00:00');
  $the_date = $date->format('Y-m-d H:i:s');

  // write first line:
  $line = [
    $data->votes->funny,
    $data->votes->useful,
    $data->votes->cool,
    $data->user_id,
    $data->review_id,
    $data->stars,
    $the_date,
    $data->business_id
  ];

  $rs = fputcsv($handle_out, $line);

  while(($json = fgets($handle)) !== false) 
  {
    $data = json_decode($json);
    $date = DateTime::createFromFormat('Y-m-d H:i:s', $data->date.' 00:00:00');
    $the_date = $date->format('Y-m-d H:i:s');

    // write first line:
    $line = [
      $data->votes->funny,
      $data->votes->useful,
      $data->votes->cool,
      $data->user_id,
      $data->review_id,
      $data->stars,
      $the_date,
      $data->business_id
    ];

    $rs = fputcsv($handle_out, $line);
  }

  fclose($handle);
  fclose($handle_out);

  $contents = file_get_contents($outfile);
}

function generate_tip_csv($infile, $outfile)
{
  $infile = str_replace(".csv", ".json", $infile);
  $handle = fopen($infile, "r");  
  $handle_out = fopen($outfile, "w"); 
  
  // First line
  $json = fgets($handle);
  $data = json_decode($json);

  // public user_id => string(22) "Vefj29mjork1DLhALLNAsg"
  // public text => string(54) "Great food, huge portions and a gift shop and showers."
  // public business_id => string(22) "JwUE5GmEO-sH1FuwJgKBlQ"
  // public likes => integer 0
  // public date => string(10) "2012-05-16"
  // public type => string(3) "tip"
    
  $attributes = ['user_id', 'business_id', 'likes', 'date'];
  $rs = fputcsv($handle_out, $attributes);

  $date = DateTime::createFromFormat('Y-m-d H:i:s', $data->date.' 00:00:00');
  $the_date = $date->format('Y-m-d H:i:s');

  // write first line:
  $line = [
    $data->user_id,
    $data->business_id,
    $data->likes,
    $the_date
  ];

  $rs = fputcsv($handle_out, $line);

  while(($json = fgets($handle)) !== false) 
  {
    $data = json_decode($json);

    $date = DateTime::createFromFormat('Y-m-d H:i:s', $data->date.' 00:00:00');
    $the_date = $date->format('Y-m-d H:i:s');
    $line = [
      $data->user_id,
      $data->business_id,
      $data->likes,
      $the_date
    ];

    $rs = fputcsv($handle_out, $line);
  }

  fclose($handle);
  fclose($handle_out);

  $contents = file_get_contents($outfile);
}

echo "<pre>starting task\n</pre>";
// generate_user_csv($infile, $outfile);
// generate_business_csv($infile, $outfile);
// generate_checkin($infile, $outfile);
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