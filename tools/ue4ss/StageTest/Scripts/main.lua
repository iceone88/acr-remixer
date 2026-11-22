-- StageTest (acr-stagegen): once the player is in a car, teleport it to the
-- points listed in ../points.txt ("label x y z yaw", Unreal cm / degrees) and
-- log "[StageTest] at <label>" after each move so the host can screenshot.
local here = debug.getinfo(1, "S").source:sub(2):gsub("\\", "/"):match("(.*/)")
local POINTS = here .. "../points.txt"
local points, state, idx, wait, done = nil, "idle", 0, 0, false
local start_loc = nil

local function load_points()
    local f = io.open(POINTS, "r")
    if not f then return nil end
    local t = {}
    for line in f:lines() do
        local lab, x, y, z, yaw = line:match("^(%S+)%s+(%S+)%s+(%S+)%s+(%S+)%s+(%S+)")
        if lab then table.insert(t, { lab, tonumber(x), tonumber(y), tonumber(z), tonumber(yaw) }) end
    end
    f:close()
    return t
end

local logged = {}
local function pawn()
    -- the car: the possessed pawn, once it is not a (service park) camera
    local pcs = FindAllOf("PlayerController") or {}
    local extra = FindAllOf("AcrPlayerController") or {}
    for _, x in ipairs(extra) do table.insert(pcs, x) end
    for _, pc in ipairs(pcs) do
        if pc:IsValid() then
            local p = pc.Pawn
            if p and p:IsValid() then
                local cls = p:GetClass():GetFName():ToString()
                if not logged[cls] then
                    logged[cls] = true
                    print(string.format("[StageTest] possessed pawn %s\n", cls))
                end
                if not string.find(cls, "Camera") then return p end
            end
        end
    end
    return nil
end

local function dist_to(p, pt)
    local l = p:K2_GetActorLocation()
    return math.sqrt((l.X - pt[2]) ^ 2 + (l.Y - pt[3]) ^ 2) / 100
end

local debug_cam = false
local function console(cmd)
    local pcs = FindAllOf("AcrPlayerController") or FindAllOf("PlayerController") or {}
    for _, pc in ipairs(pcs) do
        if pc:IsValid() then
            local ksl = StaticFindObject("/Script/Engine.Default__KismetSystemLibrary")
            ksl:ExecuteConsoleCommand(pc, cmd, pc)
            return true
        end
    end
    return false
end

-- Move the (debug) camera, not the car: teleporting the physics car crashes.
local function view_at(pt)
    if not debug_cam then
        console("ToggleDebugCamera")
        debug_cam = true
    end
    local loc = { X = pt[2], Y = pt[3], Z = pt[4] + 400 }
    local rot = { Pitch = -12, Yaw = pt[5], Roll = 0 }
    local dcc = FindFirstOf("DebugCameraController")
    if dcc and dcc:IsValid() then
        pcall(function() dcc.Pawn:K2_TeleportTo(loc, rot) end)
        pcall(function() dcc:SetControlRotation(rot) end)
    else
        console(string.format("BugItGo %.0f %.0f %.0f -12 %.1f 0", pt[2], pt[3], pt[4] + 400, pt[5]))
    end
    return 1
end

local function teleport(p, pt)
    local loc = { X = pt[2], Y = pt[3], Z = pt[4] + 150 }
    local rot = { Pitch = 0, Yaw = pt[5], Roll = 0 }
    local root = p.RootComponent
    -- K2_TeleportTo has no out-parameters (passing an FHitResult from Lua crashed UE4SS)
    local methods = {
        function() p:K2_TeleportTo(loc, rot) end,
    }
    for m, f in ipairs(methods) do
        local ok, err = pcall(f)
        local d = dist_to(p, pt)
        if d < 20 then
            pcall(function()
                root:SetPhysicsLinearVelocity({ X = 0, Y = 0, Z = 0 }, false, "None")
                root:SetPhysicsAngularVelocityInDegrees({ X = 0, Y = 0, Z = 0 }, false, "None")
            end)
            return m
        end
        print(string.format("[StageTest] method %d failed (%s), still %.0f m away\n", m, tostring(err), d))
    end
    return 0
end

StageTestAPI = {
    console = function(c) return console(c) end,
    view_at = function(pt) return view_at(pt) end,
    restart = function() done, state, start_loc, points, debug_cam = false, "idle", nil, nil, false end,
}

-- live hook: re-run ../live.lua whenever it changes (iterate without restarting the game)
local LIVE, live_src = here .. "../live.lua", ""
LoopAsync(1000, function()
    local f = io.open(LIVE, "r")
    if f then
        local src = f:read("*a"); f:close()
        if src ~= live_src then
            live_src = src
            local chunk, err = load(src, "live")
            if chunk then
                local ok, e = pcall(chunk)
                print("[StageTest] live: " .. (ok and "ok" or tostring(e)) .. "\n")
            else print("[StageTest] live load error: " .. tostring(err) .. "\n") end
        end
    end
    return false
end)

LoopAsync(1000, function()
    if done then return false end
    local ok, err = pcall(function()
        local p = pawn()
        if state == "idle" then
            -- wait until the car is on the stage and has left the start line
            points = points or load_points()
            if p and points and #points > 0 then
                local l = p:K2_GetActorLocation()
                start_loc = start_loc or l
                local moved = math.sqrt((l.X - start_loc.X) ^ 2 + (l.Y - start_loc.Y) ^ 2) / 100
                if math.abs(l.X) + math.abs(l.Y) < 1 then start_loc = nil
                elseif moved > 15 then
                    state, wait = "settle", 3
                    print(string.format("[StageTest] go: pawn %s at (%.0f, %.0f), %d points\n",
                        p:GetClass():GetFName():ToString(), l.X / 100, l.Y / 100, #points))
                end
            end
        elseif state == "settle" then
            wait = wait - 1
            if wait <= 0 then state, idx = "move", 1 end
        elseif state == "move" then
            if not p then return end
            view_at(points[idx])
            state, wait = "hold", 5
        elseif state == "hold" then
            wait = wait - 1
            if wait <= 0 then
                print(string.format("[StageTest] at %s (%.0f, %.0f, %.0f)\n", points[idx][1],
                    points[idx][2] / 100, points[idx][3] / 100, points[idx][4] / 100))
                state, wait = "shot", 3
            end
        elseif state == "shot" then
            wait = wait - 1
            if wait <= 0 then
                idx = idx + 1
                if idx > #points then
                    if debug_cam then console("ToggleDebugCamera") end
                    print("[StageTest] done\n"); done = true
                else
                    state = "move"
                end
            end
        end
    end)
    if not ok then print("[StageTest] error: " .. tostring(err) .. "\n") end
    return false
end)
print("[StageTest] loaded\n")
