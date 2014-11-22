<?php

/**
 * Phase 4: The matrix of all things was way too huge. 
 * We need a small IID correlated subset of data... 
 * so we can run SVD & RBM ML algorithms on the data... 
 */

set_time_limit(0);
require('debug.php');

function get_infile($filename)
{
  return '../data/cleaned-csv/'  . $filename;
}

function get_outfile($filename)
{
  return '../data/cleaned-csv/' . $filename;
}

// Changes to users: removed type & name
//  -elite becomes count of years they were elite instead of array of actual years
//  -friends just becomes a count of how many friends this user has (a graph representation would be expensive)
function generate_user_ratings_csv()
{
  // http://php.net/manual/en/memcached.installation.php
  // $memcache = new Memcached(); 
  // $memcache->addServer('localhost', 11211) or die ("Could not connect to memcache");

  // @todo: if we can later incorporate more data into the matrix, we will
  $user_csv = 'yelp_academic_dataset_user.csv';
  $review_csv =  'yelp_academic_dataset_review.csv';
  $business_csv = 'yelp_academic_dataset_business.csv';
  $all_user_review_csv = 'all_users_and_reviews.csv';

  // get all required file handles
  $infile_user = get_infile($user_csv);
  $infile_review = get_infile($review_csv);
  $infile_business = get_infile($business_csv);
  $outfile = get_outfile($all_user_review_csv);

  // open all files
  $handle_user = fopen($infile_user, "r");  
  $handle_review = fopen($infile_review, "r");  
  $handle_business = fopen($infile_business, "r");  
  $handle_out = fopen($outfile, "w"); 
  
  echo 'create user_attributes<br>';
  // rebuild user_attributes...
  $user_attributes = fgetcsv($handle_user, 0, ",",'"'); // first line = header
  // echo Debug::vars($user_attributes); exit;
  foreach ($user_attributes as $key => $attr)
  {
    $user_attributes[$key] = 'user.'.$user_attributes[$key];
  }
  // echo Debug::vars($user_attributes); exit;
  
  $business_attributes = fgetcsv($handle_business, 0, ",",'"'); // first line = header
  // echo Debug::vars($business_attributes); exit;
  unset($business_attributes);

  echo 'create business_hashmap<br>';
  $biz_hashes = [];
  // Create business_hashmap (add it to user_attributes)
  while(($line = fgetcsv($handle_business, 0, ",",'"')) !== false) 
  {
    $biz_hashes[] = 'business_id.'.$line[78];
  }
  shuffle($biz_hashes);

  // Now, only the first 2100x biz hashes go into our system (we ignore everything else!)
  $count = 0;
  foreach ($biz_hashes as $biz_hash)
  {
    // if($count > 2100) break;
    if($count > 200) break; // use a tiny subset at first to confirm this can work
    $user_attributes[] = $biz_hash;
    $count++;
  }
  unset($biz_hashes);
  fclose($handle_business);

  $biz_index_hashamp = [];
  // Build a business index so we can find the exact rows we need
  foreach ($user_attributes as $index => $attribute) 
  {
    if($index > 20)
    {
      $attr_parts = explode('.', $attribute);
      if($attr_parts[0] == 'business_id')
      {
        $business_hash = $attr_parts[1];
        $biz_index_hashamp[$business_hash] = $index;
      }

    }
  }
  echo Debug::vars($biz_index_hashamp); exit;

  $init_zeroes = [];
  $max = count($biz_index_hashamp);
  // initialize all extra rows to zero.
  for($i = 21; $i < $max; $i++) 
  {
    $init_zeroes[$i] = 0;
  }
  echo Debug::vars($init_zeroes); exit;

  // now populate with all user reviews
  $review_attributes = fgetcsv($handle_review, 0, ",",'"'); // first line = header
  // echo Debug::vars($review_attributes); exit;
  // 0 => string(11) "business_id"
  // 1 => string(4) "date"
  // 2 => string(9) "review_id"
  // 3 => string(5) "stars"
  // 4 => string(7) "user_id"
  
  // user-based review object
  while(($line = fgetcsv($handle_review, 0, ",",'"')) !== false) 
  {
    // Get the right column index for the business
    $biz_column_index = $biz_index_hashamp[$line[0]];
    // The user_hashkey is the yelp user_id hash
    $user_hashkey = $line[4];
    $stars = $line[3];
    // $reviews =    
  }


  // dump 42k+ attributes to file.......
  $rs = fputcsv($handle_out, $user_attributes); // first-line CSV header

  // dumping all 250k users to a massive array
  $all_users_and_reviews = [];
  echo 'building users & reviews; init zeroes <br>';
  // append all userdata with our new data
  while(($line = fgetcsv($handle_user, 0, ",",'"')) !== false) 
  {
    // initialize all users to have zero / NA for all business reviews
    $all_users_and_reviews[$line[16]] = array_merge($line, $init_zeroes);// 16 => string(7) "user_id"
  }
  fclose($handle_user);

  // Now that we have completely populated our 2D matrix with information, 
  // we can populate our CSV file with the data
  foreach ($all_users_and_reviews as $key => $line) 
  {
    $rs = fputcsv($handle_out, $line);
  }
  fclose($handle_out);

}


// echo "<pre>starting task\n</pre>";
generate_user_ratings_csv();
echo "<pre>All done! Generated: $outfile \n</pre>";