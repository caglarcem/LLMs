import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the Vicuna model and tokenizer (assumed to be hosted on Hugging Face)
model_name = "lmsys/vicuna-13b-v1.3"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 2: Define the incident sample data (ICAM analysis input)
incident_data = {
    "daniel_fischer_video_interview": """Audio file

Josh, Dan and Jake catchup-20240318_080124-Meeting Recording.mp4



Transcript

00:00:05 Speaker 1

Yeah, I've just spoken to Tim, he said. He'll. Yeah. He'll catch up with you at some stage this morning. I've been out to.

00:00:12 Speaker 1

To the rig to take photos and everything to.

00:00:15 Speaker 1

Just to have to explain the context of it, I guess to the client, when people start coming in this morning.

00:00:22 Speaker 2

OK, no problem now. Is it being managed by Stan more themselves or is a third party or?

00:00:23 Speaker 1

Right.

00:00:30 Speaker 1

They haven't said too much about it yet, but they'll have their own.

00:00:34 Speaker 1

Yeah. And reps yeah.

00:00:35 Speaker 1

You have to have to look at it.

00:00:35 Speaker 2

OK.

00:00:37 Speaker 2

No worries, Dan. How how long you been with Mitchell's man?

00:00:41 Speaker 3

I just finished my preparation period, so about six months.

00:00:45 Speaker 2

OK. And where were you before stem or?

00:00:48 Speaker 3

I I was.

00:00:51 Speaker 3

Ohh, before Stan more like.

00:00:53 Speaker 2

Yeah, yeah.

00:00:54 Speaker 3

I've did.

00:00:55 Speaker 2

Like with like with us. Like, what have you done for the six months?

00:01:00 Speaker 3

I've done about 3-3 months at 412 like it would still stand more, just a different side, yeah.

00:01:06 Speaker 2

Ohh Yep.

00:01:09 Speaker 2

That's your.

00:01:09 Speaker 3

But I understand more like I've only.

00:01:11 Speaker 3

Been.

00:01:11 Speaker 3

This is the only project I've been on.

00:01:13 Speaker 2

OK.

00:01:14

Yeah.

00:01:15 Speaker 2

Alright, no worries. Who's your UM.

00:01:17 Speaker 2

Who's are you are.

00:01:18 Speaker 2

You normally with the same crew, like, have you been with the same crew the whole six months?

00:01:23 Speaker 3

I was with Morgan when I was at Ward 12, but he got he's now a supervisor. So then I got put on cross his crew and I've been with him for about 3 months like I've done half and half with each so.

00:01:35 Speaker 2

Uh with Chucky and crossy. Yeah, OK.

00:01:39 Speaker 1

Different Morgan Morgan able.

00:01:40 Speaker 2

Oh, not not. Not a different Morgan.

00:01:42 Speaker 1

Yeah, Morgan Abel.

00:01:44 Speaker 2

Ohh yeah it is. It is Morgan able? Yeah.

00:01:50 Speaker 2

And who was the other offside?

00:01:51 Speaker 2

Are you working with?

00:01:53 Speaker 3

Josh Wilde.

00:01:55 Speaker 2

OK. Have you worked with him for a few?

00:01:56 Speaker 2

While as well like.

00:01:57 Speaker 3

Yeah, he pretty much once we got back to Troy Trail, he started that there at the same time when we got back to Troy Trail, so about 3.

00:02:05 Speaker 3

Months as well.

00:02:06 Speaker 2

OK. So have you been on that same rig the whole time?

00:02:07 Speaker 3

Yeah.

00:02:10 Speaker 3

Yeah, same rig, 1271, yeah.

00:02:12 Speaker 2

1271 alright, too easy. You just wanna like, talk me through yesterday.

00:02:22 Speaker 3

Yes. Where do you want me start? Just like just before it happened.

00:02:27 Speaker 2

Tell me about your day. Tell me wherever you, wherever you wanna start, man.

00:02:31 Speaker 3

Wow.

00:02:31 Speaker 2

You don't have to tell me what you had for breakfast or anything like that. OK? Probably probably more like.

00:02:35 Speaker 3

Well.

00:02:38 Speaker 2

Were you going back to a whole? We're going back to a.

00:02:42 Speaker 2

Hole that you.

00:02:43 Speaker 2

Had started yesterday? Or was it a new hole?

00:02:47 Speaker 3

So we pretty much we're getting ready to do A to start a full C hole like start a pre call on it, but they wanted to clean out first. So we were on that to like yesterday we finished that off and then we only got to the new hole by about two thirtyish and we will like we'll just get the.

00:03:07 Speaker 3

Casing done. Get the diverter in and just have it all ready for tomorrow. Pretty much. And that was gonna be our day.

00:03:12 Speaker 2

OK.

00:03:14 Speaker 3

And yeah, so we got there, started setting up, everything was going fine, like there wasn't any issues and.

00:03:23 Speaker 3

Pretty much it was like end of the day. Last time. We're gonna do like we finished casing off. No problems there.

00:03:29 Speaker 3

And it was just alright. We'll put the rods in, have the bit down there and just have it all ready. So tomorrow morning we just start.

00:03:37 Speaker 2

Walk in and and start. Yeah.

00:03:41 Speaker 2

OK.

00:03:42 Speaker 3

Yeah, and that was pretty much it and it was pretty much the last thing we were gonna do that day. I was he was cross. He asked me. I'll just put in the rod. Guards like the and I as I've gone to go put it in.

00:03:56 Speaker 3

My hand like got stuck in between the foot clamp holders, not the actual foot clamps themselves, but like where they are and the rod was coming down at the same time and it's like.

00:04:10 Speaker 3

The the rod guy didn't go in properly and I've gone to readjust it and as it got caught on the rod that was coming down, it's like pushed like cause I was holding on to the rod guy. It's come down my hands come down with it.

00:04:24 Speaker 3

And it's like pinched between the foot clamp holder and the rod guide.

00:04:29 Speaker 3

Kindle.

00:04:30 Speaker 3

So I like just just pinched it pretty much and I pulled out and it's yeah.

00:04:30 Speaker 2

OK.

00:04:35 Speaker 2

I just ripped.

00:04:36 Speaker 3

Yeah. Hurt like a *****.

00:04:38 Speaker 2

Yeah, I can like. Well, you can imagine like, you know you touch the back of your hand, that's sensitive, you touch the front and it's even more sensitive. So you know, you're losing that tip the, you know, see how we're designed me. And we feel something hot or feel something cold helps us move away. So yeah, I can can only imagine how much that hurt. So can you sorry.

00:04:39 Speaker 3

Alright.

00:04:53

Yeah.

00:04:56 Speaker 3

Thanks.

00:04:58 Speaker 2

You got.

00:04:59 Speaker 3

No, I think I got lucky pulling out cause I reckon if I didn't pull out earlier, I probably would have lost more. So yeah.

00:05:05 Speaker 2

Wow, dude. OK, so can you just back up and explain that bit to me again, right, so.

00:05:13 Speaker 2

There's, I guess is there is there foot clamps or rod guards? What's the what's the difference here?

00:05:19 Speaker 3

So.

00:05:20 Speaker 3

There weren't. There weren't any foot clamps in, but it's like the paths that bring the foot clamps up and down. They were like pretty much all the way up like they were out of the way.

00:05:23

Yep.

00:05:27 Speaker 2

Yep.

00:05:32 Speaker 3

But as I've like gone to throw the rod guide in, it's not landed improperly. We're just like slides in.

00:05:40 Speaker 3

And it's it looks like it got caught on the foot clamp. Like the foot clamp holders and I've like gone to readjust it just like how I do every other time. Like you, they tend to get stuck sometimes and you just gotta give it a wiggle when it falls it in all by itself, but this time.

00:05:57 Speaker 2

Yeah.

00:06:00 Speaker 3

The rod was coming down like cross. He was looking up just to get the rod to come down and I've, like, shimmied it. And obviously my shimmy it must have caught onto the rod and pulled the rod like rod guy down with it and it's then pinched onto the foot clamp holder. Like I just grabbed my finger and.

00:06:14 Speaker 2

Yep.

00:06:18

OK.

00:06:20 Speaker 3

Yeah.

00:06:21 Speaker 2

So you actually got caught between the foot clamp holder and the rod guard is that?

00:06:26 Speaker 3

Yeah, the rod guide.

00:06:27 Speaker 2

Right. Yeah. So you didn't get caught against the rod that was moving was more the Rod guard twisting and?

00:06:30 Speaker 3

No.

00:06:31 Speaker 3

It was more the volume. Yeah, it was more the rod just like pushed it down is what caused it to like.

00:06:39 Speaker 3

Because I had room for my hand to be there and it's just pushed down. And that's what's caught my finger.

00:06:44 Speaker 2

OK, no worries. And like you said, when you adjusted these in the past, you said sometimes you know they get caught or they get sticky and you gotta give them a bit of a a shovel, whatever. Is there a handle or something to hold onto with?

00:06:59 Speaker 2

Those things or.

00:06:59 Speaker 3

Well, yeah, that's.

00:07:00 Speaker 3

That's what I was holding onto. Was the handle where you just like it's out of the way. Like it's nowhere near the rod and.

00:07:04 Speaker 3

You just like shimmy it along, like go back and forth.

00:07:04

Yeah, yeah.

00:07:08 Speaker 3

And yeah, this time I guess I just got caught. Like, it's not something I was expecting to happen at all. Like I've been. I've done it like I worked at Mannion as well for six months, and I've done that. I've always put in rod's, like, these rod guides that I've never had that issue.

00:07:11 Speaker 2

OK.

00:07:24 Speaker 2

Yep, OK.

00:07:26 Speaker 3

Yeah. Yeah, it wasn't something I was expecting.

00:07:28 Speaker 3

To happen at all.

00:07:29 Speaker 2

Yeah. Yeah. Well, yeah, obviously. And so.

00:07:35 Speaker 2

He quickly pulled it out, let out a bit of a yell and let you drill and.

00:07:38 Speaker 2

Know let crossy know that you know **** it.

00:07:39 Speaker 3

Yeah, I pulled.

00:07:40 Speaker 3

Pulled it out and like instantly turned around. Removes my glove like just ripped off the glove just to make sure it was alright and that's when I just saw bone and just yeah. **** **** yeah.

00:07:51 Speaker 3

I was like ******* hell. Couldn't couldn't have just been a pinch and just like.

00:07:56 Speaker 3

Hurt, but I had to be the the.

00:07:58 Speaker 3

Yeah.

00:07:59 Speaker 3

Yeah.

00:07:59 Speaker 2

So watch. Watch this dude, did you grab, like, a rag or your shirt or?

00:08:03 Speaker 3

So Crossy yelled out, shut down like just yelled. He puts the knee stop instantly and said get to the Ute and was telling Josh get to the again the driver's seat while he went and grabbed A rag for me like a clean new rag rubbed around my finger and they just got me straight to the medic.

00:08:20 Speaker 3

So my.

00:08:21 Speaker 2

OK. How long was the drive from the rig to the medic mate?

00:08:25 Speaker 3

Ah, say it's about a 10 minute drive, maybe 15, but they were. They were going pretty quick like to get me there.

00:08:30 Speaker 2

OK.

00:08:33 Speaker 2

Yeah. They just want to get you sorted.

00:08:35 Speaker 3

Yeah, I I was. It must sound like I was in a lot, but I was. I was ******* yelling so.

00:08:41 Speaker 2

Mate, you're it.

00:08:43 Speaker 2

It's OK. Like, that's OK, you're we.

00:08:47 Speaker 2

You know, we just.

00:08:48 Speaker 2

Want to make sure you're OK? Like really?

00:08:49 Speaker 3

Yeah.

00:08:52 Speaker 2

The the the the gloves you're wearing are standard gloves that you normally would wear.

00:08:58 Speaker 3

Sorry.

00:08:58 Speaker 2

No. The gloves you were wearing nothing special. You know, you're just wearing those. The Maxi Flex or whatever they are or.

00:09:05 Speaker 3

Yeah, they were just the impact gloves, like the proper one. Yeah, impact ones. But they don't really have anything on the fingertips. They don't have it up to like the knuckles. Sort of. Yeah.

00:09:08 Speaker 2

Well, they they were the impact ones.

00:09:13 Speaker 2

On the tips.

00:09:15 Speaker 2

Yeah.

00:09:17 Speaker 2

Yeah, it's more.

00:09:17 Speaker 2

For a striking with a hammer or something drops on you.

00:09:18 Speaker 3

Yeah, it's more like if something glance here. Yeah, like on your arm.

00:09:21 Speaker 2

Yeah, rather than a that rather than a crush injury, so.

00:09:25 Speaker 2

OK. And the medic basically just cleaned you up and said take you.

00:09:30 Speaker 2

To Moranbah hospital.

00:09:32 Speaker 3

Yeah, they just had a quick look, gave it a quick claim, but because it was in the glove, they said it's pretty clean. So they were pretty happy with it and it was just give it a bandage and send me to the hospital.

00:09:44 Speaker 2

OK, so what's the what's been that outcome? Have you had a the tip of your finger amputated or is it just whilst skin or what's happened then?

00:09:52 Speaker 3

Hello.

00:09:55 Speaker 3

They've said it's.

00:09:56

It's.

00:09:56 Speaker 3

Kind of like half degloving half amputation. Like I've lost the underside of my finger is like basically all gone like the flesh is gone, but the top side of you can still see my nail. So it's kind of hard to tell.

00:10:07 Speaker 2

Yeah.

00:10:13 Speaker 3

What? What they're gonna do. We're at the Marta hospital to work out what they wanna do with it. Sort of.

00:10:20 Speaker 2

Yeah, it's. I think from our side mate, it's just making sure about like nerve damage and you know, think about the future like any arthritis concerns or stuff like that, ensuring that you still got your mobility so.

00:10:36 Speaker 3

Yeah, I could still feel the finger. So it's like, yeah.

00:10:37 Speaker 2

Yeah, like if they.

00:10:40 Speaker 2

OK. But it is it. Is it throbbing? Like is it hurting?

00:10:43 Speaker 3

Just at the end where the.

00:10:44 Speaker 3

Like the bits come off, yeah.

00:10:45 Speaker 2

Where the tip is.

00:10:46 Speaker 2

OK. And there was only that little location like right, basically on that tip, that's the only there's no bruising anywhere else or?

00:10:52 Speaker 3

It's like sort of like around here.

00:10:56

Yeah.

00:10:57 Speaker 3

And then the.

00:10:57 Speaker 3

Top side it's like a little bit cut off at the top as well. Like it sort of goes diagonal, yeah.

00:11:03 Speaker 2

So what do you recommend about a 5 cent piece, like maybe a little bit bigger?

00:11:07 Speaker 3

Just down just down to like the first.

00:11:09 Speaker 3

Knuckle. Sort of, yeah.

00:11:09 Speaker 2

OK. Yep.

00:11:12 Speaker 2

Ohh mate.

00:11:16 Speaker 2

Yeah. Is it a? Is it a design thing in your mind that that, you know, it's it's one of those things, isn't it? It's just it, no if.

00:11:23 Speaker 3

It sort of just feels like it was unfortunate, like it was.

00:11:24 Speaker 2

You got caught.

00:11:27 Speaker 3

Like no one was doing anything wrong. I've done it a million times and I think it was just bad timing with how it all just happened, like it was just unfortunate how it happened.

00:11:37 Speaker 2

Yeah, you know how cross he was. Moving the rod like, do you normally shimmy those things around while the rods moving or do you like?

00:11:38 Speaker 1

Yeah.

00:11:45 Speaker 3

Yeah, like I've, I've done it a few times where it's like that and I've like it should because the road is moving it. It should actually help you push it in, yeah.

00:11:51 Speaker 2

It's yes.

00:11:53 Speaker 2

Exactly like. So you're moving it around then?

00:11:55 Speaker 2

As it comes past with it.

00:11:56 Speaker 3

And it should just slope like just slide in because it's meant to keep the rod in place.

00:12:01 Speaker 3

Like it's it should fit in like a puzzle base, yeah.

00:12:01 Speaker 2

Yes, and and and that things and that thing's got a handle or it's got a place to hold it. So it's that doesn't require you to hold.

00:12:06 Speaker 3

Yeah.

00:12:09 Speaker 2

The rod or anything like that?

00:12:10 Speaker 3

Hey, like, you're not touching. You're just holding the handle. And you just, like, pick it up and you just drop it in. Pretty much like it should just fall in.

00:12:17 Speaker 2

Yeah. And in this case.

00:12:18 Speaker 3

So you don't have to be anywhere near.

00:12:19 Speaker 2

And in this case it fell. But like hit the the the sort of the whatever they.

00:12:24 Speaker 3

Yeah, it's like.

00:12:24 Speaker 3

Hit the guide because they were up, but they must have been just not up further enough. Like just that little bit too far in. So they're on an angle.

00:12:34 Speaker 3

Yep, yeah.

00:12:36 Speaker 2

OK. You got any questions for me? So like I I just showed your text yesterday cause I never want you to feel that you know you're alone or you got questions or what's next. So that's that's part of the reason for doing that Dan is.

00:12:49 Speaker 2

Just.

00:12:50 Speaker 2

You know, making sure everything's covered up from your side, Jake, if there's any costs or anything.

00:12:56 Speaker 2

Associated with today or visits etc.

00:12:59 Speaker 2

Or like let me know or cover it or you know we can reimbursements whatever you need, man.

00:13:02 Speaker 1

Yeah, yeah, yeah.

00:13:05 Speaker 1

Yeah. We'll talk there about that.

00:13:07 Speaker 2

And then UM.

00:13:08 Speaker 1

Like that.

00:13:09 Speaker 2

Probably probably one of the most important things today is or or if you've already got it, but have we got a work cover certificate at all?

00:13:17 Speaker 1

Yeah, we've got they've issued that at the hospital in Moranbah last night.

00:13:21 Speaker 2

Yeah. Have you a photo of it, have you?

00:13:21

We've been referrals.

00:13:23 Speaker 1

I've got it all on me so I can.

00:13:24 Speaker 1

Send everything through to you.

00:13:26 Speaker 2

OK, because I can put that through the work cover this morning. Yeah. And then I can give.

00:13:31 Speaker 2

And.

00:13:32 Speaker 2

I can give my mate a call there and see if she can push it through quickly this morning.

00:13:36 Speaker 1

Yeah, Yep. No problem. Get it. Get it moving straight away.

00:13:37

Yeah. So.

00:13:40 Speaker 2

Because if I put in, if I put it in their system, they come to work this morning and it's like ohh **** right now. So you can even just take a photo of the work cover certificate.

00:13:49 Speaker 1

Yeah.

00:13:50 Speaker 2

And just like text it to me. I'll I'll get that running this morning.

00:13:53 Speaker 1

Yeah.

00:13:55 Speaker 1

No worries.

00:13:57 Speaker 1

Alright, we're gonna go go in and.

00:14:00 Speaker 1

Say the specialist now at the matter hospital that just stopped to open up now and we'll catch up a little bit later on with with the outcome, we'll see what what their recommendations are and how we go or what we need to do from here.

00:14:05 Speaker 2

OK.

00:14:11 Speaker 3

Ours.

00:14:12 Speaker 2

Yes. So can you.

00:14:14 Speaker 2

Can you quickly do that? WCC for me? Yeah. OK, cool. Dan. You're you feel like you're being looked after and.

00:14:17 Speaker 1

Yeah, I'll send it.

00:14:18 Speaker 1

To her.

00:14:22 Speaker 3

Yeah. Yeah, I'm yeah, I'm pretty happy with how the everyone's like. It's just been working just to get me fixed up. So yeah, that's what I mainly am looking for.

00:14:28 Speaker 1

A.

00:14:33 Speaker 3

So just being out of pain, yeah.

00:14:34 Speaker 2

Yeah, I know.

00:14:36 Speaker 2

Yeah, alright. Well, you've got mine in Jakes numbers directly, and they've just spent two hours with Jake in the car anyway, so I'm. I know. So I apologise that his music tasted so ****. Alright, I'll, I'll talk to you soon, Jake. Thanks.

00:14:47 Speaker 3

Alright.

00:14:51 Speaker 2

For your help for this morning.

00:14:53 Speaker 1

Yeah, no problem. So please.

00:14:54 Speaker 2

OK. Thanks man. Bye.""",
    "event_debrief_josh_wild": """Mitchell
EVENT DEBRIEF
SERVICES
HSE-FM-31
All sections must be completed.
Name
Employment Type
Client
Site
Josh
Will
Employee [ Contractor
Stanmore
Poitrel
Who was the event reported to:
Tim
Brown
Date
17/3/24
Time:
4:46 pm
Reported to client:
Yes No
Who
Richard Heritage
Date:
17/3/24
Time:
4:58 pm
Current Role:
Drillers Assistant
Time in Role:
6 months Time in the Company:
6 months
Time in the Industry:
6 months
Event Description
Date (of event):
17/3/24
Time (of event):
4:45 am/fm
Asset ID:
1271
Exact Location of event:
Romp 30 matural area
Were you:
M Directly Involved
An Eyewitness
First on Site (Responder)
Brief description of the task being carried out at the time of the event: The injured offsider was
inserting table slips as part of set up to begin drilling. I was at
the front of the rig truck where I had moved the support
track into position to connect the 2" water hose and prepare
the fuel hose pump for refueling.
Was a Take 5 or equivalent completed before starting the task?
Yes (Please Attach)
No
O N/A
Was an equipment pre-start completed prior to starting the task?
X Yes (Please Attach)
No
N/A
How many days into the swing are you? E.g. 4th day shift out of 14
10 of 13
Did you attend a pre-start meeting at the start of the shift?
Yes O No
How would you describe the operating conditions (dusty, rough, dark etc)?
Wind gusts, some dust, hot.
and dry temprature.
How were you feeling physically after the event (aches, pains, stiffness, soreness)?
Fine .
Were you feeling fatigued or distracted (tired, headache, personal/family issues)? No.
Version:
4
Title:
Event Debrief
Uncontrolled Copy When Printed
Page 1 of 4
Date:
January 2024
Approved:
General Manager - People, Risk and Sustainability
Review by: January 2029
Scanned with CamScanner
Mitchell
EVENT DEBRIEF
SERVICES
HSE-FM-31
Step by step description leading up to and including the event (specific locations, times, tasks carried out, communications):
We had moved all equipment to the site in the
R30, natural area, set up site and finished installing the PVC
pre collor and diverter. Once the diverter was in place / moved
the support track from the rear of the vig truck to the
front of the vig truck and compressor. Once parked did stable
helped the other offsider and the driller with installing the
blooie line then went back to
to the support truck
and connected the 2" water hose from the bean pump
to the support track, I then climbed onto the room of
the track to pull out the fuel nozzle from the reel when
I heard the driller
yell "shut it down" and heard the rig
shot down. I climbed down from the truck and wort to
the deck where I saw the injured offsider so
went
to the ate to put up the tailgate and start it so we
could make our way to the ERT. After starting the vite
went back to the support truck and shout it off while
the driller shut of the compressor and Got a clean ray
for the injured offsider to hold over his finger. We then
all got in the ute and
I drove
straight to the ERT,
Was a drug and/or breathalyser test conducted?
Yes M No
Step by step description after the event occurred (locations, timeline, and communications):
After
arriving at the ERT the paramedic assisted the injured
offsider while 1 dialed "000"
to request on ambulance. We
then woited of the ERT building until the ambulance arrived
and took the injured ofsider to Moranbah hospital. After it
let
we
swiped off site, the briller had a D/A test and then
we
drove
back to camp of approximately 6:30 pm.
Version:
Title:
Event Debrief
Uncontrolled Copy When Printed
Page 2 of 4
Date:
January 2024
Approved:
General Manager - People, Risk and Sustainability
Review by: January 2029
Scanned with CamScanner
""",
    "witness_statement_richard_heritage": """stanmore
STATEMENT FOR INCIDENT FORM
STATEMENT FOR INCIDENT
Individual statements must be completed by all persons involved in or witnessing any incident / accident,
causing either injury to persons or damage to property, equipment, or the environment.
Supervisor MUST Consider:
Radio records (within 12hrs), DSS records (within12 hrs), MineStar records.
Name
RICHARD HERITAGE
Job Role
Explanation Supervisor
Department / Crew
Exploration
Title of Incident
Degloved Finger tip
Date of Incident
17/3/24
Time of Incident
16:45
Role in Incident
Incident was reported to me.
Other Possible Witnesses
I was in De office writing daily report.
I recieved a phone call from Jim Brown
Ort Daniel Fischer has injured his finger & is
being brought to the site paramedic by
Nathan Cross.
I walked over to paramedic & met up with the
drill arew.
I notified Brendon Balmain, and unidad for
Fully describe the incident
Ambulance to arrive & date Daniel to
from start to finish
(Consider actions taken,
Moranbah.
events observed, sounds
heard that may be related)
What work was being
undertaken in the time
Office work
leading up to / at the time of
the event
Note any conditions that
may have influenced the
incident (eg, Weather, dust,
time of day, equipment
No
malfunction etc)
Note any other factors that
may have contributed
(personal factors,
No
equipment conditions)
Note: Only include facts (as you know them) in your statement, not assumptions or opinions.
SMR-HSS-FRM-000006
Version Number: 1.01
Date Published: 25/05/2023
Statement for Incident Form
Document uncontrolled when downloaded or printed
Page 1 of 3
stanmore
STATEMENT FOR INCIDENT FORM
DIAGRAM
Please detail your recollection of the incident scene in a diagram as best you can. Please include reference
points/locations and "North" direction if possible.
STATEMENT DETAILS
Name of person giving
RICHARD HERITAGE
statement
Signature
Date
20/3/24
Supervisor Name
BRENDAN BALMAIN
Supervisor Signature
Dat
20/3/24
SMR-HSS-FRM-000006
Version Number: 1.01
Date Published: 25/05/2023
Statement for Incident Form
Document uncontrolled when downloaded or printed
Page 2 of 3
""",
    "event_debrief_nathan_cross": """Mitchell
EVENT DEBRIEF
SERVICES
HSE-FM-31
All sections must be completed.
Name
Employment Type
Client
Site
Nathan Cross
Employee _ Contractor
Stanmore
Poitrel
Who was the event reported to:
Tim Brown
Date: 17/3/241
Time:
4:46 pm
Reported to client:
Yes [ No
Who:
Richard Heritage Date: 17/3/24
Time:
4:58 pm
Current Role:
Driller
Time in Role:
10 yrs Time in the Company:
6 months Time in the Industry:
18 yrs
Event Description
Date (of event):
17/3/24
Time (of event):
4:45 am/pm
Asset ID:
1271
Exact Location of event:
:1 Ria table
Were you:
Directly Involved
An Eyewitness
LA First on Site (Responder)
Brief description of the task being carried out at the time of the event:
whilst lowering rod down hole with head, offsider was"
dropping in table inserts, asi
briefly looked away
the offsider want to adjust insert and caught his finger.
Was a Take 5 or equivalent completed before starting the task?
Yes (Please Attach)
No
N/A
Was an equipment pre-start completed prior to starting the task?
X Yes (Please Attach)
No
N/A
How many days into the swing are you? E.g. 4th day shift out of 14
10 of 13
Did you attend a pre-start meeting at the start of the shift?
Yes No
How would you describe the operating conditions (dusty, rough, dark etc)?
clear, warm day
How were you feeling physically after the event (aches, pains, stiffness, soreness)?
good
Were you feeling fatigued or distracted (tired, headache, personal/family issues)?
No
Version:
4
Title:
Event Debrief
Uncontrolled Copy When Printed
Page 1 of 4
Date:
January 2024
Approved:
General Manager - People, Risk and Sustainability
Review by: January 2029
Scanned with CamScanner
Mitchell
EVENT DEBRIEF
SERVICES
HSE-FM-31
Step by step description leading up to and including the event (specific locations, times, tasks carried out, communications):
After setting up diverter, one offsider went to fuel up and
Daniel and Myself went to start running in rods. I started to
lower rods with head and Daniel dropped first table
insert in, i had to stop movement for him to adjust
in place properly. While Daniel went to get second insert
started moving rods again. I happen to briefly
look down side of rig to check second offsider, I was
still lowering rods and didn'tdas Daniel dropped second
insert he had put his hand in to adjust insert into
place. This is when his finger got caught and it
was at this time he screamed. We engaged 'E' stops
shutting down all equipment. Second offsider quickly got
ute ready to
go as
1 got clean rag wrapped around
Daniel's finger and we proceeded to medics. I called
supervisor (Tim Brown) on way to medics who met us
down at medics office
incident occured at 16:45
supervisor notified at 16:46
arrived at medics at 17:00
ambulance calle
Nord at 17:10
and arrived approx 17:50
Was a drug and/or breathalyser test conducted?
Yes No
Step by step description after the event occurred (locations, timeline, and communications):
Dan got seen by medics, Tim notified stanmore supervisors
approx
16:58 who then notified who they had to. Daniel
went in ambulance
a DRA at 18:20 before is
at appox 18:00, and 1 completed
leaving site to go to
camp.
Version:
4
Title:
Event Debrief
Uncontrolled Copy When Printed
Page 2 of 4
Date:
January 2024
Approved:
General Manager - People, Risk and Sustainability
Review by: January 2029
Scanned with CamScanner
""",
    "witness_statement_timothy_brown": """stanmore
STATEMENT FOR INCIDENT FORM
STATEMENT FOR INCIDENT
Individual statements must be completed by all persons involved in or witnessing any incident / accident,
causing either injury to persons or damage to property, equipment, or the environment.
Supervisor MUST Consider:
Radio records (within 12hrs), DSS records (within12 hrs), MineStar records
Name
TimoTHH
BROWN
MITCHELL DELL
Job Role
SUPERVISOR
Department / Crew
EXPLORATION
Title of Incident
PARTIAL AMPUTATION OF LEFT HAND
RING FINGER
Date of Incident
17/03/24
Time of Incident
4:45 Pm
Role in Incident
PERSON REPORTED TO BY CIREN
Other Possible Witnesses
SHOWING DA FROM RIG 1270 WATER
POINT @ ORICA YARD $ FILL PROCESS.
SAN TEXT MESSAGE & MISSED CALL
FROM DRILLER, CALLED DRILLER BACK $
WAS ADVISED OF INCIDENT.
CONTACTED STANMORE APPOINTED SUPERVISOR
TO REPORT INCIDENT & ADVISE CREW
WERE ON THEIR WAY TO THE MEDIC .
I THEN LEFT WATER POINT TO GO TO
Fully describe the incident
from start to finish
MEDIC & CHECK SITUATION/ WORKER.
(Consider actions taken,
events observed, sounds
heard that may be related)
What work was being
SHOWING DA FROM OTHER RIG WATER
undertaken in the time
leading up to / at the time of
POINT & FILLING PROCESS
the event
Note any conditions that
may have influenced the
END OF SHIFT.
incident (eg, Weather, dust,
time of day, equipment
malfunction etc)
Note any other factors that
DRILLER DISTRACTED BY 2ND DA
may have contributed
(personal factors,
IP ATTEMPTED TO FINISH TASK WHILE
equipment conditions)
DRILLER DISTRACTED
Note: Only include facts (as you know them) in your statement, not assumptions or opinions.
SMR-HSS-FRM-000006
Version Number: 1.01
Date Published: 25/05/2023
Statement for Incident Form
Document uncontrolled when downloaded or printed
Page 1 of 3
stanmore
STATEMENT FOR INCIDENT FORM
DIAGRAM
Please detail your recollection of the incident scene in a diagram as best you can. Please include reference
points/locations and "North" direction if possible.
NOT PRESENT ONSITE.
STATEMENT DETAILS
Name of person giving
TIMOTHY BROWN
statement
Signature
Dat
20/03/2024
Supervisor Name
BRENDAN BALMAIN
Supervisor Signature
Dbali
Dat
20/05/2024
SMR-HSS-FRM-000006
Version Number: 1.01
Date Published: 25/05/2023
Statement for Incident Form
Document uncontrolled when downloaded or printed
Page 2 of 3
"""
}

# Combine all the data into a single string for analysis
combined_incident_data = f"""
Follow up interview: {incident_data['daniel_fischer_video_interview']}

Event Debrief-1: {incident_data['event_debrief_josh_wild']}

Event Debrief-2: {incident_data['event_debrief_nathan_cross']}

Witness Statement-1: {incident_data['witness_statement_richard_heritage']}

Witness Statement-2: {incident_data['witness_statement_timothy_brown']}

Please analyze the inputs provided in this session and identify all potential contributing factors. For each contributing factor, list the corresponding actions to address it and provide a detailed description of each action.
"""

# Tokenize the input text
inputs = tokenizer(combined_incident_data, return_tensors="pt")

# Generate text from the model
output = model.generate(inputs['input_ids'], max_length=15000, max_new_tokens=15000, num_beams=4, temperature=0.7, early_stopping=True)

# Decode and print the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Vicuna Analysis:")
print(generated_text)
