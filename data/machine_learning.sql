SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";
CREATE DATABASE IF NOT EXISTS `machine_learning` DEFAULT CHARACTER SET utf8 COLLATE utf8_general_ci;
USE `machine_learning`;

DROP TABLE IF EXISTS `business`;
CREATE TABLE IF NOT EXISTS `business` (
  `id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `type` varchar(10) DEFAULT 'business',
  `business_id` varchar(255) DEFAULT NULL,
  `name` varchar(255) DEFAULT NULL,
  `neighborhoods` text,
  `full_address` varchar(255) DEFAULT NULL,
  `city` varchar(255) DEFAULT NULL,
  `state` varchar(255) DEFAULT NULL,
  `latitude` decimal(12,8) DEFAULT NULL,
  `longitude` decimal(12,8) DEFAULT NULL,
  `stars` varchar(10) DEFAULT NULL,
  `review_count` int(10) DEFAULT NULL,
  `categories` text,
  `open` enum('True','False') DEFAULT NULL,
  `hours` text,
  `attributes` text,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 AUTO_INCREMENT=1 ;

DROP TABLE IF EXISTS `check-in`;
CREATE TABLE IF NOT EXISTS `check-in` (
  `id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `type` varchar(7) DEFAULT 'checkin',
  `business_id` varchar(255) DEFAULT NULL,
  `checkin_info` mediumtext,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 AUTO_INCREMENT=1 ;

DROP TABLE IF EXISTS `review`;
CREATE TABLE IF NOT EXISTS `review` (
  `id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `type` varchar(7) DEFAULT 'review',
  `business_id` varchar(255) DEFAULT NULL,
  `user_id` varchar(255) DEFAULT NULL,
  `stars` decimal(6,4) DEFAULT NULL,
  `text` text,
  `date` timestamp NULL DEFAULT NULL,
  `votes` mediumtext,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 AUTO_INCREMENT=1 ;

DROP TABLE IF EXISTS `tip`;
CREATE TABLE IF NOT EXISTS `tip` (
  `id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `type` varchar(10) DEFAULT 'tip',
  `text` text,
  `business_id` varchar(255) DEFAULT NULL,
  `user_id` varchar(255) DEFAULT NULL,
  `date` timestamp NULL DEFAULT NULL,
  `likes` int(10) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 AUTO_INCREMENT=1 ;

DROP TABLE IF EXISTS `user`;
CREATE TABLE IF NOT EXISTS `user` (
  `id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `user_id` varchar(255) DEFAULT NULL,
  `type` varchar(6) DEFAULT 'user',
  `name` varchar(100) DEFAULT NULL,
  `review_count` int(10) DEFAULT NULL,
  `average_stars` decimal(6,4) DEFAULT NULL,
  `votes` mediumtext,
  `friends` mediumtext,
  `elite` decimal(6,4) DEFAULT NULL,
  `yelping_since` timestamp NULL DEFAULT NULL,
  `compliments` mediumtext,
  `fans` int(10) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 AUTO_INCREMENT=1 ;
