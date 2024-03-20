from django.shortcuts import redirect


def pre_handle_request(get_response):
    def middleware(request):
        # 사용자가 요청한 경로
        uri = request.get_full_path()

        if 'accounts' not in uri and 'oauth' not in uri and 'api' not in uri and 'static' not in uri\
                and 'admin' not in uri and 'upload' not in uri:
            if 'join' not in uri and 'login' not in uri and uri != '/' and 'terms' not in uri \
                    and 'company' not in uri and 'club/intro' not in uri\
                    and 'notice' not in uri and 'festival' not in uri \
                    and 'pr-post-list' not in uri and 'teenplay' not in uri and 'wishlist' not in uri:
                if request.session.get('member') is None:
                    request.session['previous_uri'] = uri
                    return redirect('/member/login')

            if request.user_agent.is_mobile:
                if 'app' not in uri:
                    uri = f'/app{uri}'
                    return redirect(uri)

            else:
                if 'app' in uri:
                    uri = uri.replace('/app', '')
                    return redirect(uri)

        # if 'admin' in uri:
        #     if 'login' not in uri:
        #         if request.session.get('admin') is None:
        #             request.session['previous_uri'] = uri
        #             return redirect('/admin/login/')

        response = get_response(request)

        return response

    return middleware

